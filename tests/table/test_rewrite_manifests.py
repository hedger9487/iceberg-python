# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

from pathlib import PosixPath

import pyarrow as pa
import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.catalog.memory import InMemoryCatalog
from pyiceberg.exceptions import NoSuchTableError, ValidationException
from pyiceberg.manifest import ManifestEntryStatus, ManifestFile
from pyiceberg.schema import Schema
from pyiceberg.table.snapshots import Operation
from pyiceberg.types import IntegerType, NestedField, StringType


@pytest.fixture
def catalog(tmp_path: PosixPath) -> InMemoryCatalog:
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=tmp_path.absolute().as_posix())
    catalog.create_namespace("default")
    return catalog


def _drop_table(catalog: Catalog, identifier: str) -> None:
    try:
        catalog.drop_table(identifier)
    except NoSuchTableError:
        pass


def test_rewrite_manifests_empty_table_fails(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_empty_table"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    with pytest.raises(ValidationException, match="Cannot rewrite manifests on a table with no snapshots"):
        tbl.rewrite_manifests().commit()


def test_rewrite_manifests_basic(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_basic"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 2, "data": "b"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 3, "data": "c"}], schema=arrow_schema))

    current_snap = tbl.current_snapshot()
    assert current_snap is not None
    initial_manifests = current_snap.manifests(tbl.io)
    assert len(initial_manifests) == 3

    initial_records = tbl.scan().to_arrow().to_pylist()
    assert len(initial_records) == 3

    tbl.rewrite_manifests().commit()

    rewritten_snap = tbl.current_snapshot()
    assert rewritten_snap is not None
    assert rewritten_snap.summary is not None
    assert rewritten_snap.summary.operation == Operation.REPLACE
    assert rewritten_snap.snapshot_id != current_snap.snapshot_id
    assert rewritten_snap.parent_snapshot_id == current_snap.snapshot_id

    rewritten_records = tbl.scan().to_arrow().to_pylist()
    assert rewritten_records == initial_records
    assert rewritten_snap.summary["manifests-created"] == "1"
    assert rewritten_snap.summary["manifests-replaced"] == "3"
    assert rewritten_snap.summary["entries-processed"] == "3"
    assert rewritten_snap.summary["manifests-kept"] == "0"


def test_rewrite_manifests_via_maintenance(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_via_maintenance"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 2, "data": "b"}], schema=arrow_schema))

    current_snap = tbl.current_snapshot()
    assert current_snap is not None
    assert len(current_snap.manifests(tbl.io)) == 2

    tbl.maintenance.rewrite_manifests().commit()

    snap = tbl.current_snapshot()
    assert snap is not None
    assert snap.summary is not None
    assert snap.summary.operation == Operation.REPLACE
    assert len(snap.manifests(tbl.io)) == 1
    assert sorted(tbl.scan().to_arrow().to_pylist(), key=lambda r: r["id"]) == [{"id": 1, "data": "a"}, {"id": 2, "data": "b"}]


def test_rewrite_manifests_preserves_sequence_numbers(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_preserves_sequence_numbers"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema, properties={"format-version": "2"})

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 2, "data": "b"}], schema=arrow_schema))

    snap2 = tbl.current_snapshot()
    assert snap2 is not None
    assert snap2.sequence_number == 2

    tbl.rewrite_manifests().commit()

    snap3 = tbl.current_snapshot()
    assert snap3 is not None
    assert snap3.sequence_number == 3

    new_manifests = snap3.manifests(tbl.io)
    assert len(new_manifests) == 1

    entries = list(new_manifests[0].fetch_manifest_entry(tbl.io, discard_deleted=False))
    assert len(entries) == 2

    for entry in entries:
        assert entry.status == ManifestEntryStatus.EXISTING

    entry_seq_numbers = {entry.data_file.file_path: entry.sequence_number for entry in entries}
    assert 1 in entry_seq_numbers.values()
    assert 2 in entry_seq_numbers.values()


def test_rewrite_manifests_with_predicate(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_with_predicate"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))
    snap1 = tbl.current_snapshot()
    assert snap1 is not None

    tbl.append(pa.Table.from_pylist([{"id": 2, "data": "b"}], schema=arrow_schema))
    snap2 = tbl.current_snapshot()
    assert snap2 is not None

    tbl.append(pa.Table.from_pylist([{"id": 3, "data": "c"}], schema=arrow_schema))
    snap3 = tbl.current_snapshot()
    assert snap3 is not None

    manifests = snap3.manifests(tbl.io)
    assert len(manifests) == 3

    # Rewrite only manifests added by snap1 or snap2
    target_manifest_paths = {m.manifest_path for m in manifests if m.added_snapshot_id in {snap1.snapshot_id, snap2.snapshot_id}}

    tbl.rewrite_manifests().rewrite_if(lambda m: m.manifest_path in target_manifest_paths).commit()

    snap4 = tbl.current_snapshot()
    assert snap4 is not None
    new_manifests = snap4.manifests(tbl.io)
    # The 2 targeted manifests were merged into 1, while 1 was kept as-is -> total 2 manifests
    assert len(new_manifests) == 2
    assert sorted(tbl.scan().to_arrow().to_pylist(), key=lambda r: r["id"]) == [
        {"id": 1, "data": "a"},
        {"id": 2, "data": "b"},
        {"id": 3, "data": "c"},
    ]


def test_rewrite_manifests_branch(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_branch"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 2, "data": "b"}], schema=arrow_schema))

    initial_snap = tbl.current_snapshot()
    assert initial_snap is not None

    branch = "feature_branch"
    tbl.manage_snapshots().create_branch(snapshot_id=initial_snap.snapshot_id, branch_name=branch).commit()

    tbl.rewrite_manifests(branch=branch).commit()

    current_snap = tbl.current_snapshot()
    assert current_snap is not None
    assert current_snap.snapshot_id == initial_snap.snapshot_id
    assert len(current_snap.manifests(tbl.io)) == 2

    branch_snap = tbl.snapshot_by_name(branch)
    assert branch_snap is not None
    assert branch_snap.snapshot_id != initial_snap.snapshot_id
    assert len(branch_snap.manifests(tbl.io)) == 1
    assert sorted(tbl.scan().use_ref(branch).to_arrow().to_pylist(), key=lambda r: r["id"]) == [
        {"id": 1, "data": "a"},
        {"id": 2, "data": "b"},
    ]


def test_rewrite_manifests_in_transaction(catalog: Catalog) -> None:
    identifier = "default.test_rewrite_manifests_in_transaction"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 2, "data": "b"}], schema=arrow_schema))

    with tbl.transaction() as tx:
        tx.rewrite_manifests().commit()

    snap = tbl.current_snapshot()
    assert snap is not None
    assert snap.summary is not None
    assert snap.summary.operation == Operation.REPLACE
    assert len(snap.manifests(tbl.io)) == 1
    assert sorted(tbl.scan().to_arrow().to_pylist(), key=lambda r: r["id"]) == [{"id": 1, "data": "a"}, {"id": 2, "data": "b"}]


def test_rewrite_manifests_partitioned_table(catalog: Catalog) -> None:
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    identifier = "default.test_rewrite_manifests_partitioned_table"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "category", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="category"))
    tbl = catalog.create_table(identifier, schema=schema, partition_spec=spec)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("category", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "category": "cat_a"}, {"id": 2, "category": "cat_b"}], schema=arrow_schema))
    tbl.append(pa.Table.from_pylist([{"id": 3, "category": "cat_a"}], schema=arrow_schema))

    current_snap = tbl.current_snapshot()
    assert current_snap is not None
    assert len(current_snap.manifests(tbl.io)) == 2

    tbl.rewrite_manifests().commit()

    snap = tbl.current_snapshot()
    assert snap is not None
    assert snap.summary is not None
    assert snap.summary.operation == Operation.REPLACE
    assert len(snap.manifests(tbl.io)) == 1

    entries = list(snap.manifests(tbl.io)[0].fetch_manifest_entry(tbl.io))
    assert len(entries) == 3
    partitions = {entry.data_file.partition[0] for entry in entries}
    assert partitions == {"cat_a", "cat_b"}
    assert sorted(tbl.scan().to_arrow().to_pylist(), key=lambda r: r["id"]) == [
        {"id": 1, "category": "cat_a"},
        {"id": 2, "category": "cat_b"},
        {"id": 3, "category": "cat_a"},
    ]


def test_rewrite_manifests_file_count_mismatch_raises(catalog: Catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    identifier = "default.test_rewrite_manifests_file_count_mismatch"
    _drop_table(catalog, identifier)
    schema = Schema(
        NestedField(1, "id", IntegerType(), required=True),
        NestedField(2, "data", StringType(), required=False),
    )
    tbl = catalog.create_table(identifier, schema=schema)

    arrow_schema = pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False),
            pa.field("data", pa.string(), nullable=True),
        ]
    )

    tbl.append(pa.Table.from_pylist([{"id": 1, "data": "a"}], schema=arrow_schema))

    from pyiceberg.table.update.snapshot import RewriteManifests

    orig_create_manifest = RewriteManifests._create_manifest

    def buggy_create_manifest(
        self: RewriteManifests, spec_id: int, manifest_bin: list[ManifestFile]
    ) -> tuple[ManifestFile | None, int]:
        manifest, count = orig_create_manifest(self, spec_id, manifest_bin)
        assert manifest is not None
        # Artificially alter existing_files_count (index 8) to simulate file count mismatch
        manifest._data[8] = (manifest._data[8] or 0) + 1
        return manifest, count

    monkeypatch.setattr(RewriteManifests, "_create_manifest", buggy_create_manifest)

    with pytest.raises(ValidationException, match="Replaced and created manifests must have the same number of active files"):
        tbl.rewrite_manifests().commit()

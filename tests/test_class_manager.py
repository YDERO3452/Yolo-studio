"""Tests for core/class_manager.py."""

import json
from pathlib import Path

import pytest

from core.class_manager import COCO_EN_ZH_MAP, ClassManager


class TestClassManagerBasics:
    """Tests for ClassManager initialization and basic properties."""

    def test_init_empty_project(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.classes == []
        assert mgr.get_class_count() == 0
        assert len(mgr) == 0

    def test_init_with_existing_classes_txt(self, tmp_path):
        (tmp_path / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.classes == ["person", "car"]
        assert len(mgr) == 2

    def test_init_with_existing_colors(self, tmp_path):
        (tmp_path / "classes.txt").write_text("person\n", encoding="utf-8")
        (tmp_path / "classes.colors").write_text(
            json.dumps({"person": [255, 0, 0]}), encoding="utf-8"
        )
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.get_class_color("person") == (255, 0, 0)

    def test_init_defaults_to_cwd(self):
        mgr = ClassManager()
        assert isinstance(mgr.project_dir, Path)

    def test_init_creates_project_dir(self, tmp_path):
        new_dir = tmp_path / "new_project"
        assert not new_dir.exists()
        ClassManager(project_dir=str(new_dir))
        assert new_dir.exists()

    def test_init_generates_missing_colors(self, tmp_path):
        (tmp_path / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.get_class_color("person") is not None
        assert mgr.get_class_color("car") is not None

    def test_empty_project_is_falsy_len(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert len(mgr) == 0


class TestClassCRUD:
    """Tests for add/remove/rename operations."""

    def test_add_class_success(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        result = mgr.add_class("person")
        assert result is True
        assert "person" in mgr.classes
        assert mgr.get_class_count() == 1

    def test_add_class_duplicate_returns_false(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        result = mgr.add_class("person")
        assert result is False
        assert len(mgr.classes) == 1

    def test_add_class_generates_color(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        color = mgr.get_class_color("person")
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_remove_class_success(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        result = mgr.remove_class("person")
        assert result is True
        assert "person" not in mgr.classes

    def test_remove_class_not_found(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        result = mgr.remove_class("nonexistent")
        assert result is False

    def test_remove_class_also_removes_color(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        mgr.remove_class("person")
        assert mgr.get_class_color("person") is None

    def test_rename_class_success(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        result = mgr.rename_class("person", "human")
        assert result is True
        assert mgr.classes == ["human"]

    def test_rename_class_preserves_color(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        orig_color = mgr.get_class_color("person")
        mgr.rename_class("person", "human")
        assert mgr.get_class_color("human") == orig_color

    def test_rename_class_old_not_found(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        result = mgr.rename_class("nonexistent", "newname")
        assert result is False

    def test_rename_class_new_already_exists(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        mgr.add_class("human")
        result = mgr.rename_class("person", "human")
        assert result is False


class TestColorManagement:
    """Tests for class color operations."""

    def test_get_class_color_existing(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        color = mgr.get_class_color("person")
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_get_class_color_missing(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.get_class_color("nonexistent") is None

    def test_set_class_color_success(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        result = mgr.set_class_color("person", (0, 255, 0))
        assert result is True
        assert mgr.get_class_color("person") == (0, 255, 0)

    def test_set_class_color_class_not_found(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        result = mgr.set_class_color("nonexistent", (0, 255, 0))
        assert result is False

    def test_get_color_alias(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("car")
        assert mgr.get_color("car") == mgr.get_class_color("car")

    def test_set_color_alias(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("car")
        mgr.set_color("car", (128, 128, 128))
        assert mgr.get_class_color("car") == (128, 128, 128)

    def test_colors_are_distinct_for_different_classes(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("A")
        mgr.add_class("B")
        mgr.add_class("C")
        cA = mgr.get_class_color("A")
        cB = mgr.get_class_color("B")
        cC = mgr.get_class_color("C")
        assert cA != cB
        assert cA != cC
        assert cB != cC


class TestIndexLookups:
    """Tests for class index/name lookups."""

    def test_get_class_index_found(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        mgr.add_class("car")
        assert mgr.get_class_index("car") == 1

    def test_get_class_index_not_found(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.get_class_index("nonexistent") is None

    def test_get_class_by_index_valid(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        assert mgr.get_class_by_index(0) == "person"

    def test_get_class_by_index_out_of_range(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.get_class_by_index(999) is None
        assert mgr.get_class_by_index(-1) is None

    def test_get_class_name_alias(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("dog")
        assert mgr.get_class_name(0) == "dog"

    def test_get_class_id_alias(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("cat")
        assert mgr.get_class_id("cat") == 0

    def test_get_or_create_existing(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        idx = mgr.get_or_create_class("person")
        assert idx == 0
        assert mgr.get_class_count() == 1

    def test_get_or_create_new(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        idx = mgr.get_or_create_class("new_class")
        assert idx == 0
        assert "new_class" in mgr.classes

    def test_get_all_classes_returns_copy(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        classes = mgr.get_all_classes()
        classes.append("tainted")
        assert "tainted" not in mgr.classes


class TestNameMapping:
    """Tests for class name mapping (model → project)."""

    def test_init_loads_default_coco_map(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert "person" in mgr.name_map
        assert mgr.name_map["person"] == "人"

    def test_map_class_name_with_mapping(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.map_class_name("person") == "人"

    def test_map_class_name_without_mapping(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.map_class_name("unknown_thing") == "unknown_thing"

    def test_set_name_mapping(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.set_name_mapping("vehicle", "车辆")
        assert mgr.map_class_name("vehicle") == "车辆"

    def test_remove_name_mapping(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.remove_name_mapping("person")
        assert mgr.map_class_name("person") == "person"

    def test_get_name_map_returns_copy(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        nm = mgr.get_name_map()
        nm["tainted"] = "bad"
        assert "tainted" not in mgr.name_map

    def test_import_model_names_with_translate(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        # Clear the default COCO map so we can test import fresh
        mgr.name_map = {}
        added = mgr.import_model_names({0: "person", 1: "car", 2: "alien"})
        assert added == 3
        assert mgr.map_class_name("person") == "人"
        assert mgr.map_class_name("car") == "汽车"
        assert mgr.map_class_name("alien") == "alien"  # No COCO mapping

    def test_import_model_names_no_translate(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.name_map = {}
        added = mgr.import_model_names({0: "person"}, translate=False)
        assert added == 1
        assert mgr.map_class_name("person") == "person"

    def test_import_model_names_skips_existing(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.name_map = {"person": "人类"}
        added = mgr.import_model_names({0: "person"})
        assert added == 0
        assert mgr.map_class_name("person") == "人类"

    def test_load_name_map_from_file(self, tmp_path):
        (tmp_path / "class_name_map.json").write_text(
            json.dumps({"dog": "狗"}), encoding="utf-8"
        )
        mgr = ClassManager(project_dir=str(tmp_path))
        assert mgr.map_class_name("dog") == "狗"


class TestImportFromList:
    """Tests for import_from_list."""

    def test_import_new_classes(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.import_from_list(["person", "car", "bike"])
        assert mgr.classes == ["person", "car", "bike"]
        assert mgr.get_class_count() == 3

    def test_import_skips_existing(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        mgr.import_from_list(["person", "car"])
        assert mgr.classes == ["person", "car"]

    def test_import_generates_colors(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.import_from_list(["person", "car"])
        assert mgr.get_class_color("person") is not None
        assert mgr.get_class_color("car") is not None


class TestSaveLoad:
    """Tests for persistence roundtrip."""

    def test_save_creates_files(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        mgr.save()
        assert (tmp_path / "classes.txt").exists()
        assert (tmp_path / "classes.colors").exists()
        assert (tmp_path / "class_name_map.json").exists()

    def test_roundtrip_save_and_reload(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        mgr.add_class("person")
        mgr.add_class("car")
        mgr.set_class_color("person", (255, 0, 0))
        mgr.set_name_mapping("vehicle", "车辆")
        mgr.save()

        mgr2 = ClassManager(project_dir=str(tmp_path))
        assert mgr2.classes == ["person", "car"]
        assert mgr2.get_class_color("person") == (255, 0, 0)
        assert mgr2.map_class_name("vehicle") == "车辆"

    def test_save_handles_empty_classes(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        result = mgr.save()
        assert result is True
        assert (tmp_path / "classes.txt").exists()

    def test_save_returns_false_on_error(self, tmp_path):
        mgr = ClassManager(project_dir=str(tmp_path))
        # Make the classes file path a directory to cause write failure
        (tmp_path / "classes.txt").mkdir()
        mgr.add_class("person")
        result = mgr.save()
        assert result is False


class TestReadWriteHelpers:
    """Tests for static read/write helper methods."""

    def test_read_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        lines = ClassManager.read_lines(str(f))
        assert lines == ["line1", "line2", "line3"]

    def test_read_lines_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        lines = ClassManager.read_lines(str(f))
        assert lines == [""] or lines == []

    def test_read_json(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps({"a": 1, "b": [2, 3]}), encoding="utf-8")
        data = ClassManager.read_json(str(f))
        assert data == {"a": 1, "b": [2, 3]}

    def test_save_json(self, tmp_path):
        f = tmp_path / "out.json"
        ClassManager.save_json({"key": "val"}, str(f))
        loaded = json.loads(f.read_text(encoding="utf-8"))
        assert loaded == {"key": "val"}

    def test_save_json_handles_non_ascii(self, tmp_path):
        f = tmp_path / "cn.json"
        ClassManager.save_json({"name": "中文"}, str(f), ensure_ascii=False)
        loaded = json.loads(f.read_text(encoding="utf-8"))
        assert loaded["name"] == "中文"

"""Tests for core/project_manager.py — ProjectManager."""

import json
import pytest
from pathlib import Path

from core.project_manager import ProjectManager


class TestProjectManagerInit:
    """Tests for ProjectManager initialization."""

    def test_init_creates_workspace(self, tmp_path):
        workspace = tmp_path / "workspace"
        pm = ProjectManager(str(workspace))
        assert workspace.exists()
        assert pm.workspace_root == workspace


class TestSlugify:
    """Tests for _slugify helper."""

    def test_basic_slug(self):
        pm = ProjectManager("/tmp/test")
        slug = pm._slugify("My Project")
        assert isinstance(slug, str)
        assert len(slug) > 0
        assert " " not in slug

    def test_chinese_slug(self):
        pm = ProjectManager("/tmp/test")
        slug = pm._slugify("我的项目")
        assert isinstance(slug, str)
        assert len(slug) > 0

    def test_special_chars(self):
        pm = ProjectManager("/tmp/test")
        slug = pm._slugify("test@#$%project!!!")
        assert "@" not in slug
        assert "#" not in slug

    def test_empty_name(self):
        pm = ProjectManager("/tmp/test")
        slug = pm._slugify("")
        assert len(slug) > 0


class TestCreateProject:
    """Tests for create_project method."""

    def test_creates_directory_structure(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("Test Project")
        root = Path(project["root"])
        assert root.exists()
        assert (root / "images").exists()
        assert (root / "labels").exists()
        assert (root / "models").exists()
        assert (root / "runs").exists()

    def test_creates_project_json(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("Test Project")
        root = Path(project["root"])
        project_json = root / "project.json"
        assert project_json.exists()
        data = json.loads(project_json.read_text(encoding="utf-8"))
        assert data["name"] == "Test Project"

    def test_creates_classes_file(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("Test", classes=["person", "car"])
        root = Path(project["root"])
        classes_file = root / "classes.txt"
        assert classes_file.exists()
        content = classes_file.read_text(encoding="utf-8")
        assert "person" in content
        assert "car" in content

    def test_default_classes(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("Test")
        root = Path(project["root"])
        classes_file = root / "classes.txt"
        content = classes_file.read_text(encoding="utf-8")
        assert "目标" in content

    def test_project_has_timestamp(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("Test")
        assert "created_at" in project
        assert "updated_at" in project


class TestListProjects:
    """Tests for list_projects method."""

    def test_empty_workspace(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        projects = pm.list_projects()
        assert projects == []

    def test_lists_created_projects(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        pm.create_project("Project A")
        pm.create_project("Project B")
        projects = pm.list_projects()
        assert len(projects) == 2
        names = {p["name"] for p in projects}
        assert "Project A" in names
        assert "Project B" in names

    def test_discovers_manual_projects(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        manual_dir = tmp_path / "manual_project"
        manual_dir.mkdir()
        (manual_dir / "project.json").write_text(
            json.dumps({"name": "Manual", "root": str(manual_dir)}),
            encoding="utf-8",
        )
        projects = pm.list_projects()
        assert any(p["name"] == "Manual" for p in projects)

    def test_removes_nonexistent_projects(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("To Delete")
        root = Path(project["root"])

        import shutil
        shutil.rmtree(root)

        projects = pm.list_projects()
        assert len(projects) == 0


class TestDeleteProject:
    """Tests for delete_project method."""

    def test_deletes_files(self, tmp_path):
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("To Delete")
        root = Path(project["root"])
        assert root.exists()
        pm.delete_project(project, delete_files=True)
        assert not root.exists()
        # After deleting files, list_projects should not find it
        projects = pm.list_projects()
        assert len(projects) == 0

    def test_keeps_files_by_default(self, tmp_path):
        """By default, delete_project only removes from registry, keeps files."""
        pm = ProjectManager(str(tmp_path))
        project = pm.create_project("To Keep")
        root = Path(project["root"])
        pm.delete_project(project, delete_files=False)
        # Files still exist
        assert root.exists()
        # But project.json still exists so it gets re-discovered
        projects = pm.list_projects()
        assert any(p["name"] == "To Keep" for p in projects)

    def test_nonexistent_project(self, tmp_path):
        """Deleting nonexistent project doesn't raise."""
        pm = ProjectManager(str(tmp_path))
        pm.delete_project({"root": "/nonexistent/path", "name": "fake"})

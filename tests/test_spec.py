import copy

import pytest

from crowdsim.configio import ConfigError, load_yaml, stadium_path
from crowdsim.stadium.spec import parse_stadium_spec


@pytest.fixture()
def football_dict():
    return load_yaml(stadium_path("football_bowl"))


def _parse(d):
    return parse_stadium_spec(d, key="test", source="test.yaml")


def test_preset_parses(football_dict):
    spec = _parse(football_dict)
    assert [t.id for t in spec.tiers] == ["lower", "upper"]
    assert spec.concourse("level1").z == 10.0


def test_unknown_key_is_rejected(football_dict):
    d = copy.deepcopy(football_dict)
    d["tiers"][0]["rowz"] = 3
    with pytest.raises(ConfigError, match="unknown key"):
        _parse(d)


def test_missing_key_is_rejected(football_dict):
    d = copy.deepcopy(football_dict)
    del d["tiers"][0]["rows"]
    with pytest.raises(ConfigError, match="missing required key 'rows'"):
        _parse(d)


def test_duplicate_level_ids_rejected(football_dict):
    d = copy.deepcopy(football_dict)
    d["concourses"][1]["id"] = "upper"
    d["tiers"][1]["vomitories"]["concourse"] = "upper"
    d["concourses"][1]["connectors"][0]["to"] = "ground"
    with pytest.raises(ConfigError, match="duplicate level ids"):
        _parse(d)


def test_cross_aisle_row_out_of_range(football_dict):
    d = copy.deepcopy(football_dict)
    d["tiers"][0]["cross_aisle"]["row"] = 999
    with pytest.raises(ConfigError, match="row"):
        _parse(d)


def test_stairs_must_go_down(football_dict):
    d = copy.deepcopy(football_dict)
    d["concourses"][1]["z"] = -5.0
    with pytest.raises(ConfigError, match="downwards"):
        _parse(d)


def test_vomitory_to_unknown_concourse(football_dict):
    d = copy.deepcopy(football_dict)
    d["tiers"][0]["vomitories"]["concourse"] = "nowhere"
    with pytest.raises(ConfigError, match="unknown concourse"):
        _parse(d)


def test_bad_number_type(football_dict):
    d = copy.deepcopy(football_dict)
    d["seat_width"] = "wide"
    with pytest.raises(ConfigError, match="expected a number"):
        _parse(d)

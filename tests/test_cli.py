from crowdsim.cli import main


def test_list(capsys):
    assert main(["list"]) == 0
    assert "football_bowl" in capsys.readouterr().out


def test_stadium_command(tmp_path, capsys):
    assert main(["stadium", "indoor_arena", "--out", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("indoor_arena_*.png"))) == 3


def test_crowd_command(capsys):
    assert main(["crowd", "indoor_arena", "panic", "--seed", "1"]) == 0
    assert '"agents"' in capsys.readouterr().out


def test_unknown_preset_gives_config_error(capsys):
    assert main(["stadium", "moon_base", "--no-plots"]) == 2
    assert "Unknown stadium preset" in capsys.readouterr().out

from omegaconf import OmegaConf

from utils.config_schema import apply_structured_schema


def test_wb_entity_default_is_dashlab():
    cfg = OmegaConf.create({})
    normalized = apply_structured_schema(cfg)

    assert normalized.wb.entity == "dashlab"
    assert normalized.wb.project == "autofl-testing"


def test_wb_entity_can_be_overridden():
    cfg = OmegaConf.create(
        {
            "wb": {
                "entity": "my-team",
                "project": "my-project",
                "mode": "offline",
            }
        }
    )
    normalized = apply_structured_schema(cfg)

    assert normalized.wb.entity == "my-team"
    assert normalized.wb.project == "my-project"
    assert normalized.wb.mode == "offline"

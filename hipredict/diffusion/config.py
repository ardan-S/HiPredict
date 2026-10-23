from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from hipredict.paths import files_dir

# Global IR10 statistics the released diffusion models were trained with.
IR10_GLOBAL_CENTER = 0.103687465
IR10_GLOBAL_SCALE = 0.04842728

Direction = Literal["ir2red", "red2ir", "bidirectional"]
EdgeMode = Literal["sobel", "dexined", "soft_canny", "none"]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class NormalizationConfig(_Config):
    """The affine (x - center) / scale applied to IR10 and RED4, then clipped to ±clamp."""

    center: float = IR10_GLOBAL_CENTER
    scale: float = Field(IR10_GLOBAL_SCALE, gt=0)
    clamp: Optional[float] = Field(5.0, gt=0)


class DataConfig(_Config):
    # path to csv file containing the list of images
    csv_path: str = Field(
        default_factory=lambda: str(files_dir() / "data_record_alls.csv")
    )

    # path to directory containing the observations; only used for records whose
    # paths sit under 'npy_files_all' (relative paths resolve against the CSV)
    data_path: str = Field(default_factory=lambda: str(files_dir() / "npy_files_all"))

    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class ModelConfig(_Config):
    in_channels: int = Field(3, gt=0)
    out_channels: int = Field(1, gt=0)
    base_channels: int = Field(128, gt=0)
    num_res_blocks: int = Field(3, gt=0)
    dropout: float = Field(0.0, ge=0, lt=1)
    direction: Direction = "ir2red"
    # Preprocessing the model was trained with; stored with every checkpoint.
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class DDPMConfig(ModelConfig):
    num_timesteps: int = Field(1000, gt=0)
    beta_start: float = Field(1e-4, gt=0)
    beta_end: float = Field(1e-2, gt=0)
    edge_mode: EdgeMode = "sobel"
    type: Literal["UnidirectionalDDPM", "BidirectionalDDPM"] = "UnidirectionalDDPM"


class FMConfig(ModelConfig):
    edge_mode: EdgeMode = "sobel"
    type: Literal["UnidirectionalFM", "BidirectionalFM"] = "UnidirectionalFM"
    Mode: Literal["Bridge", "Condition"] = "Bridge"


class TrainingConfig(_Config):
    batch_size: int = Field(16, gt=0)
    num_workers: int = Field(4, ge=0)
    val_num_workers: int = Field(2, ge=0)
    seed: int = 42

    # Optimiser parameters
    lr: float = Field(1e-4, gt=0)
    lr_decay: float = Field(0.9, gt=0, le=1)
    lr_decay_every: int = Field(5_000, gt=0)
    total_steps: int = Field(100_000, gt=0)

    # Weight of the signed Sobel loss on the FM velocity field
    lambda_grad: float = Field(0.0, ge=0)

    # EMA parameters
    use_ema: bool = True
    ema_decay: float = Field(0.9999, gt=0, lt=1)

    # Save settings
    save_every: int = Field(10_000, gt=0)
    log_every: int = Field(1000, gt=0)
    val_every: int = Field(10_000, gt=0)
    save_dir: str = "outputs/models/"
    resume: bool = False


class InferenceConfig(_Config):
    # Model parameters
    batch_size: int = Field(4, gt=0)
    num_workers: int = Field(8, ge=0)
    ckpt_path: str = "outputs/models/latest.pt"

    # Output
    output_dir: str = "outputs/inference/"

    direction: Direction = "ir2red"

    # Unused (None) for DDPM checkpoints
    FM_ODE_steps: Optional[int] = Field(10, gt=0)

    FM_solver: Literal["euler", "rk4"] = "euler"

    # After each FM ODE step, anchor the clean estimate x1_pred's mean to the correct mean
    dc_anchor: bool = True

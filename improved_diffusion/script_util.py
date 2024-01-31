from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel



def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    timestep_respacing,
    dataset):

    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        dropout=dropout,
        dataset=dataset
    )
    diffusion = create_gaussian_diffusion(diffusion_steps=diffusion_steps, timestep_respacing=timestep_respacing)
    return model, diffusion


def create_model(image_size,num_channels,num_res_blocks,attention_resolutions,num_heads,num_heads_upsample,dropout, dataset):
    if image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    color_dim = 1 if dataset in ['mnist','triangles'] else 3
    return UNetModel(
        in_channels=color_dim,
        model_channels=num_channels,
        out_channels= color_dim*2,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
    )


def create_gaussian_diffusion(diffusion_steps, timestep_respacing):
    betas = gd.get_named_beta_schedule(num_diffusion_timesteps=diffusion_steps)
    loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON),
        model_var_type=(gd.ModelVarType.LEARNED_RANGE),
        loss_type= loss_type)


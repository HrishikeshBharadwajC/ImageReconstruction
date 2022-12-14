hyperparameters_WGAN = {
    "epochs": 3000,
    "batch_size": 64,
    "gp_weight": 10,
    "critic_iters": 5,
    "num_workers": 2,
    "learning_rate_discriminator": 1e-4,
    "learning_rate_generator": 5e-5,
    "save_every_n_epochs": 50,
    "dataset_path": "../data/facades",
    "load_model_from_epoch": 2651,
    "dataset_direction": "reverse",
}

hyperparameters_CycleGAN = {
    "epochs": 200,
    "batch_size": 1,
    "lambda_x":10,
    "lambda_y":10,
    "num_workers": 2,
    "learning_rate_discriminator": 2e-4,
    "generator_lr_step_size":100,
    "epochs_initial_lr":100,
    "discriminator_lr_step_size":100,
    "learning_rate_generator": 2e-4,
    "save_every_n_epochs": 10,
    "dataset_path": "../data/cityscapes",
    "load_model_from_epoch": 0,
    "dataset_direction": "reverse",
}

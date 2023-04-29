import torch
import wandb

def save_model_weights(
            model, 
            model_checkpoint_path, 
            save_as_artifact, 
            artifact_name,
            model_validation_score
        ):
    # Get the model state
    model_state = model.state_dict()

    # Save the best model
    torch.save(model_state, model_checkpoint_path)

    # Also save the final model as an artifact
    if save_as_artifact:
        artifact = wandb.Artifact(artifact_name, type='model', metadata={'Validation score': model_validation_score})
        artifact.add_file(model_checkpoint_path)
        wandb.run.log_artifact(artifact) # type: ignore
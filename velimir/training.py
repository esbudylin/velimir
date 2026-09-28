import copy
import logging


def train_model(model, train_func, eval_func, scheduler, max_epochs, patience):
    best_validation_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_func()
        validation_loss = eval_func()
        scheduler.step(validation_loss)

        logging.info(
            f"Epoch {epoch} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )

        if validation_loss + 1e-5 < best_validation_loss:
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            best_validation_loss = validation_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered at epoch %d", epoch)
                break

    return best_state_dict, best_validation_loss, epoch + 1

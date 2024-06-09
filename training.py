def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # Set the model to training mode - important for batch normalization and dropout layers # Unnecessary in this situation but added for best practices
    model.train()

    for num_batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if num_batch % 100 == 0:
            loss, current = loss.item(), num_batch * batch_size + len(X)

            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

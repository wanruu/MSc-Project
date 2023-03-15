import torch
import datetime
from tqdm import tqdm


def train(dataloader, model, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        total_loss = 0.0
        for data, labels in tqdm(dataloader):
            mask = data.gt(0)
            loss = model(data, token_type_ids=None, attention_mask=mask, labels=labels)[0]
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss = total_loss / len(data)
        print(f"Epoch: {epoch}, loss: {total_loss:.2f}")
    
    cur_time = str(datetime.datetime.now())
    torch.save(model.state_dict(), cur_time + ".pt")
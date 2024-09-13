import torch
import torch.nn.functional as F
from configs import args


def test_cv(data_source, model, classes, processor):
    model.eval()
    total_loss = 0
    correct = 0
    text_inputs = [f"This is a photo of a {label}" for label in classes]
    data_iterator = data_source
    num_data = 0.0
    for batch_id, batch in enumerate(data_iterator):
        data, targets = batch
        clip_inputs = processor(
            text=text_inputs,
            images=data,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        ).to(args.device)
        targets = targets.to(args.device)

        outputs = model(**clip_inputs)
        logits_per_image = outputs.logits_per_image
        total_loss += F.cross_entropy(logits_per_image, targets, reduction="sum").item()
        probs = logits_per_image.softmax(dim=-1)
        pred = probs.argmax(dim=-1)
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        num_data += len(data)
        # print(f"target: {targets}, pred: {pred}")

    acc = float(correct) / float(num_data)
    total_l = total_loss / float(num_data)

    model.train()
    return total_l, acc


def test_poison_cv(
    helper, data_source, model, classes, processor, adversarial_index=-1
):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    data_iterator = data_source
    num_data = 0.0
    text_inputs = [f"This is a photo of a {label}" for label in classes]

    for batch_id, batch in enumerate(data_iterator):

        if args.attack_mode.lower() in ["mr", "dba"]:
            data, target = helper.get_poison_batch(batch, evaluation=True)
        elif args.attack_mode.lower() == "combine":
            data, target = helper.get_poison_batch(
                batch, evaluation=True, adversarial_index=adversarial_index
            )
        else:
            for pos in range(len(batch[0])):
                batch[1][pos] = helper.params["poison_label_swap"]

            data, target = batch

        clip_inputs = processor(
            text=text_inputs,
            images=data,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
        ).to(args.device)

        target = target.to(args.device)

        outputs = model(**clip_inputs)
        logits_per_image = outputs.logits_per_image
        total_loss += F.cross_entropy(logits_per_image, target, reduction="sum").item()
        probs = logits_per_image.softmax(dim=-1)
        pred = probs.argmax(dim=-1)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        num_data += len(data)

        # print(f"target: {target}, pred: {pred}")

    acc = float(correct) / float(num_data)
    total_l = total_loss / float(num_data)
    model.train()
    return total_l, acc

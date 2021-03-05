#!/usr/bin/env python3
"""Recipe for training a recognizer on kids' speech.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/ogi

Authors
 * Peter Plantinga 2021
"""
import sys
import torch
import speechbrain as sb
from ogi_prepare import prepare_ogi
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Adding time-domain SpecAugment if specified
        if hasattr(self.hparams, "augmentation") and stage == sb.Stage.TRAIN:
            wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        out = self.modules.recognizer(feats)
        out = self.modules.recognizer_output(out)
        logprobs = self.hparams.log_softmax(out)

        return logprobs

    def compute_objectives(self, predictions, batch, stage):
        wavs, wav_lens = batch.sig
        phonemes, phoneme_lens = batch.tokens

        if stage != sb.Stage.TRAIN:
            decoded = sb.decoders.ctc_greedy_decode(
                predictions, wav_lens, blank_id=self.encoder.get_blank_index()
            )
            self.per_metrics.append(
                ids=batch.id,
                predict=decoded,
                target=phonemes,
                target_len=phoneme_lens,
                ind2lab=self.encoder.decode_ndim,
            )

        ctc_loss = sb.nnet.losses.ctc_loss(
            log_probs=predictions,
            targets=phonemes,
            input_lens=wav_lens,
            target_lens=phoneme_lens,
            blank_index=self.encoder.get_blank_index(),
        )
        return ctc_loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            stage_stats = {
                "loss": stage_loss,
                "PER": per,
            }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            epoch_stats = {"epoch": epoch, "lr": old_lr}
            self.hparams.train_logger.log_stats(
                epoch_stats, {"loss": self.train_loss}, stage_stats
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["PER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.test_stats_file, "w") as w:
                self.per_metrics.write_stats(w)


def dataio_prep(hparams):
    """Creates datasets and loading pipelines"""

    encoder = sb.dataio.encoder.CTCTextEncoder(blank_label="<blank>")

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("phonemes")
    @sb.utils.data_pipeline.provides("tokens_list", "tokens")
    def text_pipeline(phonemes):
        tokens_list = phonemes.strip().split()
        yield tokens_list
        tokens = encoder.encode_sequence(tokens_list)
        yield torch.LongTensor(tokens)

    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "tokens"],
        ).filtered_sorted(sort_key="length")

    encoder.update_from_didataset(data["train"], output_key="tokens_list")

    return data, encoder


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    run_on_main(
        prepare_ogi,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_file": hparams["train_annotation"],
            "valid_file": hparams["valid_annotation"],
            "test_file": hparams["test_annotation"],
        },
    )

    datasets, encoder = dataio_prep(hparams)

    # Create brain object for training
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.encoder = encoder

    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="PER",
        test_loader_kwargs=hparams["dataloader_opts"],
    )

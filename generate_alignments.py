#!/usr/bin/env python3
"""Script for generating alignments from model trained with alignment loss.

To run this script, do the following:
> python generate_alignments.py hparams/train.yaml --data_folder /path/to/ogi

Authors
 * Peter Plantinga 2021
"""
import sys
import json
import torch
import speechbrain as sb
from ogi_prepare import prepare_ogi
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

STEP_SIZE = 0.04


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        out = self.modules.recognizer(feats)
        logits = self.modules.recognizer_output(out)

        return logits

    def generate_alignments(self, dataset, outputfile):
        "When training with alignment loss is finished, write alignments to file"
        alignments = {}
        dataset = dataset.filtered_sorted(key_min_value={"length": 4.0})
        dataloader = self.make_dataloader(dataset, sb.Stage.VALID, None)
        for batch in dataloader:
            logit_batch = self.compute_forward(batch, sb.Stage.VALID)
            for uttid, logits, words, length in zip(
                batch.id, logit_batch, batch.words_encoded, batch.sig.lengths
            ):
                if len(words) > 1:
                    alignment = self.get_alignment(logits, words, length)
                    alignments[uttid] = alignment

        with open(outputfile, "w") as w:
            json.dump(alignments, w, indent=2)

    def get_alignment(self, logits, words, length):
        "Returns list of starting and stopping times, one per word"

        # Only want alignments for multi-word phrases
        if len(words) == 1:
            return None

        # Get edit distance alignment between reference and predictions
        reference = [p for word in words for p in word]
        prediction = sb.decoders.ctc_greedy_decode(
            logits.unsqueeze(0), length.unsqueeze(0), self.encoder.get_blank_index()
        )[0]
        op_table = sb.utils.edit_distance.op_table(reference, prediction)
        error_rate_alignment = sb.utils.edit_distance.alignment(op_table)

        # Find word starting and stopping indexes/times
        reference_indexes = list(reference_borders(words))
        prediction_times = list(
            greedy_ctc_decode_with_times(logits, length, self.encoder.get_blank_index())
        )
        prev_stop = 0

        # Iterate alignment and record times
        starts = []
        stops = []
        for symbol, i, j in error_rate_alignment:

            # Ignore insertions
            if i is None:
                continue

            # Determine if this is a word for recording the time
            phoneme, border = reference_indexes[i]
            if border in ["start", "stop", "only"]:

                # If an important phoneme was deleted, use the prev phoneme time
                start_time, stop_time = prev_stop, prev_stop
                if j is not None:
                    start_time, stop_time, phoneme = prediction_times[j]
                    prev_stop = stop_time

                # "only" records both start and stop times
                if border == "start" or border == "only":
                    starts.append(start_time)
                if border == "stop" or border == "only":
                    stops.append(stop_time)

        alignment = {
            "starts": " ".join("{:.2f}".format(t) for t in starts),
            "stops": " ".join("{:.2f}".format(t) for t in stops),
        }

        return alignment


def reference_borders(words):
    """For each phoneme, returns 'start', 'stop', or 'only'
    to indicate that it begins/ends a word.
    
    Yields
    ------
    tuple
        (phoneme, one of "start", "stop", or "only")
    """
    for word in words:
        if len(word) == 1:
            yield (word[0], "only")
        else:
            yield (word[0], "start")
            for phoneme in word[1:-1]:
                yield (phoneme, ".")
            yield (word[-1], "stop")


def greedy_ctc_decode_with_times(logits, length, blank_index=0):
    """Yields (start_time, stop_time, phoneme) decoded from logits.

    Arguments
    ---------
    logits : torch.tensor
        The outputs of the phoneme recognizer.
    """
    length = int(len(logits) * length)
    predicted_phonemes = torch.argmax(logits, dim=-1)[:length]
    prev_phoneme = blank_index

    # Iterate logits to find distinct phonemes
    start_time = 0.
    for i, phoneme in enumerate(predicted_phonemes):
        if phoneme != prev_phoneme:
            stop_time = (i + 1) * STEP_SIZE
            if prev_phoneme != blank_index:
                yield (start_time, stop_time, prev_phoneme)
            prev_phoneme = phoneme
            start_time = stop_time

    # Append last phoneme
    if prev_phoneme != blank_index:
        stop_time = (i + 1) * STEP_SIZE
        yield (start_time, stop_time, prev_phoneme)


def dataio_prep(hparams):
    """Creates datasets and loading pipelines"""

    encoder = sb.dataio.encoder.CTCTextEncoder()
    encoder.add_blank()

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("phonemes")
    @sb.utils.data_pipeline.provides("phonemes_list", "words_encoded")
    def text_pipeline(phonemes):
        word_list = phonemes.strip().split()
        phonemes_list = [p for word in word_list for p in word.split(".")]
        yield phonemes_list
        words_encoded = [encoder.encode_sequence(w.split(".")) for w in word_list]
        yield words_encoded

    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "words_encoded"],
        ).filtered_sorted(sort_key="length")

    encoder.load_or_create(
        hparams["encoder_save_file"],
        from_didatasets=[data["train"]],
        output_key="phonemes_list",
        sequence_input=True,
    )

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

    # Recover best model and record alignments
    hparams["checkpointer"].recover_if_possible(min_key="PER")
    asr_brain.generate_alignments(datasets["train"], hparams["alignment_file"])
    asr_brain.generate_alignments(datasets["valid"], hparams["valid_alignment_file"])

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.mel_utils import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()


        # print("--------------------------------------------------")
        # print("batch[true_wav].shape:", batch["true_wav"].shape)
        # print("batch[mel_spec].shape:", batch["mel_spec"].shape)
        # print("--------------------------------------------------")

        outputs = self.model(**batch)
        batch.update(outputs)

        # print("--------------------------------------------------")
        # print("wav_generated shape:", batch["model_output"]["wav_generated"].shape)
        # print("wav_generated", batch["model_output"]["wav_generated"][0])
        # print("batch[model_output][disc_gen][msd][outputs].shape", batch["model_output"]["disc_gen"]["msd"]["outputs"][0].shape)
        # print("batch[model_output][disc_gen][msd][feature_maps].shape", batch["model_output"]["disc_gen"]["msd"]["feature_maps"][0].shape)

        # print("batch[model_output][disc_gen][mpd][outputs].shape", batch["model_output"]["disc_gen"]["mpd"]["outputs"][0].shape)
        # print("batch[model_output][disc_gen][mpd][feature_maps].shape", batch["model_output"]["disc_gen"]["mpd"]["feature_maps"][0].shape)

        # print("batch[model_output][disc_true][msd][outputs].shape", batch["model_output"]["disc_true"]["msd"]["outputs"][0].shape)
        # print("batch[model_output][disc_true][msd][feature_maps].shape", batch["model_output"]["disc_true"]["msd"]["feature_maps"][0].shape)

        # print("batch[model_output][disc_true][mpd][outputs].shape", batch["model_output"]["disc_true"]["mpd"]["outputs"][0].shape)
        # print("batch[model_output][disc_true][mpd][feature_maps].shape", batch["model_output"]["disc_true"]["mpd"]["feature_maps"][0].shape)
        # print("LEN:", len(batch["model_output"]["disc_true"]["mpd"]["feature_maps"]))
        # print("--------------------------------------------------")

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        # print("--------------------------------------------------")
        # print("loss:", batch["loss"].item())
        # print("gen_loss:", batch["gen_loss"].item())
        # print("disc_loss:", batch["disc_loss"].item())
        # print("gen_adv_loss:", batch["gen_adv_loss"].item())
        # print("fm_loss:", batch["fm_loss"].item())
        # print("mel_loss:", batch["mel_loss"].item())
        
        # print("--------------------------------------------------")

        if self.is_train:
            batch["gen_loss"].backward(retain_graph=True)

            batch["disc_loss"].backward()


            self._clip_grad_norm()
            self.generator_optimizer.step()
            self.discriminator_optimizer.step()

            # batch["loss"].backward()  # sum of all losses is always called loss
            # self._clip_grad_norm()
            # self.optimizer.step()
            if self.generator_scheduler is not None:
                self.generator_scheduler.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            mel_spec_conf = MelSpectrogramConfig()
            mel_transform = MelSpectrogram(mel_spec_conf).to("cuda")
            num_examples = 3
            for ind in range(num_examples):
                if "wav_generated" in batch["model_output"]:
                    self.writer.add_audio(
                        f"{mode}/wav_generated_{ind}",
                        batch["model_output"]["wav_generated"][ind].cpu(),
                        sample_rate=22050,  # Replace with your actual sampling rate
                    )

                    gen_mel_spec = mel_transform(batch["model_output"]["wav_generated"])[ind].cpu().detach().numpy()

                    # Convert mel spectrogram to an image-like format for logging
                    self.writer.add_image(
                        f"{mode}/gen_mel_spec_{ind}",
                        gen_mel_spec,  # Ensure it's in the correct shape (e.g., [C, H, W])
                    )

                if "true_wav" in batch:
                    self.writer.add_audio(
                        f"{mode}/true_wav_{ind}",
                        batch["true_wav"][ind].cpu(),
                        sample_rate=22050,  # Replace with your actual sampling rate
                    )

                if "mel_spec" in batch:
                    mel_spec = batch["mel_spec"][ind].cpu().numpy()

                    # Convert mel spectrogram to an image-like format for logging
                    self.writer.add_image(
                        f"{mode}/mel_spec_{ind}",
                        mel_spec,  # Ensure it's in the correct shape (e.g., [C, H, W])
                    )

        else:
            # Log Stuff
            pass

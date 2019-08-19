import logging
import tensorflow as tf
import typing
import numpy as np

from delira.models.abstract_network import AbstractTfNetwork
tf.keras.backend.set_image_data_format('channels_first')
logger = logging.getLogger(__name__)

#from delira.models.gan.alt_p2p_Discr import Discriminator
#from delira.models.gan.alt_p2p_Generator import Generator
#from delira.models.gan.bicycle_encoder import ResNet, ResNet18
from delira.models.gan.cgan_models import ResNet18, Discriminator, Discriminator_zz, Generator, Generator_add_all


class BicycleGenerativeAdversarialNetworkBaseTf(AbstractTfNetwork):
    """Implementation of Bicycle GAN for multimodal image to image translation based on https://arxiv.org/abs/1711.11586


        Notes
        -----


        References
        ----------
        https://github.com/junyanz/BicycleGAN
        https://github.com/gitlimlab/BicycleGAN-Tensorflow

        See Also
        --------
        :class:`AbstractTfNetwork`

    """
    def __init__(self, image_size: int, input_c_dim: int, output_c_dim: int, df_dim: int,
                 gf_dim: int, latent_dim: int, n_latent: int, coeff_gan: int, coeff_vae: int, coeff_reconstruct: int, coeff_latent:int,
                 coeff_kl: int, coeff_ds: int, f_size: int, is_batchnorm: bool, with_DS: bool, is_condD: bool, **kwargs):
        """

        Constructs graph containing model definition and forward pass behavior

        """
        # register params by passing them as kwargs to parent class __init__
        super().__init__(image_size=image_size, input_c_dim=input_c_dim, output_c_dim=output_c_dim, df_dim=df_dim, gf_dim=gf_dim,
                         latent_dim=latent_dim, n_latent=n_latent, coeff_gan=coeff_gan, coeff_vae=coeff_vae, coeff_reconstruct=coeff_reconstruct,
                         coeff_latent=coeff_latent, coeff_kl=coeff_kl, coeff_ds=coeff_ds, f_size=f_size, is_batchnorm=is_batchnorm, with_DS=with_DS,
                         is_condD=is_condD, **kwargs)

        is_bicycle = True
        self.with_DS = with_DS

        self.coeff_gan = coeff_gan
        self.coeff_vae = coeff_vae
        self.coeff_reconstruct = coeff_reconstruct
        self.coeff_latent = coeff_latent
        self.coeff_kl = coeff_kl
        self.coeff_ds = coeff_ds
        self.latent_dim = latent_dim
        self.n_latent = n_latent

        real_data = tf.placeholder(shape=[None, input_c_dim + output_c_dim, image_size, image_size], dtype=tf.float32)
        self.image_a = real_data[:, :input_c_dim, :, :]
        self.image_b = real_data[:, input_c_dim:input_c_dim + output_c_dim, :, :]
        self.z = tf.placeholder(shape=[None, latent_dim], dtype=tf.float32)

        self.discr, self.discr2, self.gen, self.enc = self._build_models(output_c_dim, df_dim, gf_dim, latent_dim,
                                                                         f_size, is_batchnorm, is_bicycle)
        # conditional VAE-GAN: B -> z -> B'
        z_encoded, z_encoded_mu, z_encoded_log_sigma = self.enc(self.image_b)
        image_ab_encoded = self.gen(self.image_a, z_encoded)

        # conditional Latent Regressor-GAN: z -> B' -> z'
        image_ab = self.gen(self.image_a, self.z)  # image_ab is equivalent to fake B
        z_recon, z_recon_mu, z_recon_log_sigma = self.enc(image_ab)

        # Discriminate real/fake images

        if is_condD:

            # Discriminator is used here in conditional setting

            discr_ip_ab = tf.concat([self.image_a, self.image_b], 1)
            D_real = self.discr(discr_ip_ab)  # For real image_b
            discr_ip_ab = tf.concat([self.image_a, image_ab_encoded], 1)
            D_fake_encoded = self.discr(discr_ip_ab)  # For encoded fake image_ab
            discr_ip_ab = tf.concat([self.image_a, image_ab], 1)
            D_fake = self.discr2(discr_ip_ab)  # For fake image_ab


        else:

            # Discriminator is used here in unconditional setting

            D_real = self.discr(self.image_b)
            D_fake_encoded = self.discr(image_ab_encoded)
            D_fake = self.discr2(image_ab)


        self.inputs = [real_data, self.z]
        self.outputs_train = [z_recon_mu, z_encoded_mu, z_encoded_log_sigma, D_real, D_fake, D_fake_encoded, image_ab, image_ab_encoded]
        self.outputs_eval = [z_recon_mu, z_encoded_mu, z_encoded_log_sigma, D_real, D_fake, D_fake_encoded, image_ab, image_ab_encoded]



        for key, value in kwargs.items():
            setattr(self, key, value)

    def _add_losses(self, losses: dict):
        """
        Adds losses to model that are to be used by optimizers or during evaluation

        Parameters
        ----------
        losses : dict
            dictionary containing all losses. Individual losses are averaged for discr_real, discr_fake and gen
        """
        if self._losses is not None and len(losses) != 0:
            logging.warning('Change of losses is not yet supported')
            raise NotImplementedError()
        elif self._losses is not None and len(losses) == 0:
            pass
        else:
            self._losses = {}


            # Loss function according to TF implementation

            loss_vae_gan = []
            loss_gan = []
            loss_image_cycle = []
            loss_latent_cycle = []

            total_loss = []

            for name, _loss in losses.items():
                if name == 'L2':
                    loss_val = tf.reduce_mean(_loss(self.outputs_train[3], tf.ones_like(self.outputs_train[3]))) + \
                               tf.reduce_mean(_loss(self.outputs_train[5], tf.zeros_like(self.outputs_train[5])))
                    loss_vae_gan.append(loss_val)
                    loss_val = tf.reduce_mean(_loss(self.outputs_train[3], tf.ones_like(self.outputs_train[3]))) +\
                               tf.reduce_mean(_loss(self.outputs_train[4], tf.zeros_like(self.outputs_train[4])))
                    loss_gan.append(loss_val)


            loss_vae_gan = tf.reduce_mean(loss_vae_gan, axis=0)
            self._losses['loss_vae_gan'] = loss_vae_gan

            loss_gan = tf.reduce_mean(loss_gan, axis=0)
            self._losses['loss_gan'] = loss_gan

            for name, _loss in losses.items():
                if name == 'L1':
                    loss_val = tf.reduce_mean(_loss(self.image_b, self.outputs_train[7]))
                    loss_image_cycle.append(loss_val)
                    loss_val = tf.reduce_mean(_loss(self.z, self.outputs_train[0]))
                    loss_latent_cycle.append(loss_val)

            # Image reconstruction loss in cVAE_GAN
            loss_image_cycle = tf.reduce_mean(loss_image_cycle, axis=0)
            self._losses['loss_image_cycle'] = loss_image_cycle

            # DS GAN regularisor
            loss_ds_gan = tf.divide(tf.reduce_mean(tf.abs(self.outputs_train[7] - self.outputs_train[6])), tf.reduce_mean(tf.abs(self.z - self.outputs_train[1])))
            #loss_ds_gan = tf.minimum(loss_ds_gan, -1.0)
            self._losses['ds_gan_regularisor'] = loss_ds_gan

            # Latent value reconstruction loss
            loss_latent_cycle = tf.reduce_mean(loss_latent_cycle, axis=0)
            self._losses['loss_latent_cycle'] = loss_latent_cycle

            # KL loss
            loss_kl = - 0.5 * tf.reduce_mean(1 + 2 * self.outputs_train[2] - self.outputs_train[1] ** 2 - tf.exp(2 * self.outputs_train[2]))
            self._losses['loss_kl'] = loss_kl

            if self.with_DS:
                loss_val = self.coeff_vae * loss_vae_gan - self.coeff_reconstruct * loss_image_cycle + loss_gan * self.coeff_gan - \
                           self.coeff_latent * (
                               loss_latent_cycle) - self.coeff_kl * loss_kl - self.coeff_ds * loss_ds_gan
                self._losses['total_loss'] = loss_val
                total_loss.append(loss_val)
            else:
                loss_val = self.coeff_vae * loss_vae_gan - self.coeff_reconstruct * loss_image_cycle + loss_gan * self.coeff_gan - \
                           self.coeff_latent * (loss_latent_cycle) - self.coeff_kl * loss_kl
                self._losses['total_loss'] = loss_val
                total_loss.append(loss_val)


            self.outputs_train.append(self._losses)
            self.outputs_eval.append(self._losses)

    def _add_optims(self, optims: dict):
        """
        Adds optims to model that are to be used by optimizers or during training

        Parameters
        ----------
        optim: dict
            dictionary containing all optimizers, optimizers should be of Type[tf.train.Optimizer]
        """
        if self._optims is not None and len(optims) != 0:
            logging.warning('Change of optims is not yet supported')
            pass
        elif self._optims is not None and len(optims) == 0:
            pass
        else:
            self._optims = optims

            # Update generator weights
            optim_gen = self._optims['G']
            grads_gen = optim_gen.compute_gradients(-self._losses['total_loss'], var_list=self.gen.trainable_variables)
            steps_gen = optim_gen.apply_gradients(grads_gen)

            # Update encoder weights
            optim_enc = self._optims['E']
            grads_enc = optim_enc.compute_gradients(-self._losses['total_loss'], var_list=self.enc.trainable_variables)
            steps_enc = optim_enc.apply_gradients(grads_enc)

            # Update discriminator weights
            optim_discr = self._optims['D']
            grads_discr = optim_discr.compute_gradients(self._losses['total_loss'], var_list=[self.discr.trainable_variables,
                                                                                           self.discr2.trainable_variables])
            step_discr = optim_discr.apply_gradients(grads_discr)

            steps = tf.group([steps_gen, steps_enc, step_discr])

            self.outputs_train.append(steps)

    @staticmethod
    def _build_models(output_c_dim, df_dim, gf_dim, latent_dim, f_size, is_batchnorm, is_bicycle):
        """
        builds generator and discriminators

        Parameters
        ----------

        Returns
        -------
        tf.keras.Sequential
            created gen
        tf.keras.Sequential
            created discr
        """

        discr = Discriminator(df_dim, f_size, 3, is_batchnorm, is_bicycle)
        discr2 = Discriminator(df_dim, f_size, 4, is_batchnorm, is_bicycle)

        """
        #Generator used for adding latent values only at the geerator input
        gen = Generator(output_c_dim, gf_dim, f_size, latent_dim, is_batchnorm, is_bicycle)
        
        """
        # Generator used for adding latent values at all the input layers
        gen = Generator_add_all(output_c_dim, gf_dim, f_size, latent_dim, is_batchnorm, is_bicycle)

        enc = ResNet18(df_dim, latent_dim, f_size, is_batchnorm)

        # A separate discriminator for latent values
        #discr_z = Discriminator_zz()

        return discr, discr2, gen, enc



    @staticmethod
    def closure(model: typing.Type[AbstractTfNetwork], data_dict: dict,
                metrics={}, fold=0, **kwargs):
        """
                closure method to do a single prediction.
                This is followed by backpropagation or not based state of
                on model.train

                Parameters
                ----------
                model: AbstractTfNetwork
                    AbstractTfNetwork or its child-clases
                data_dict : dict
                    dictionary containing the data
                metrics : dict
                    dict holding the metrics to calculate
                fold : int
                    Current Fold in Crossvalidation (default: 0)
                **kwargs:
                    additional keyword arguments

                Returns
                -------
                dict
                    Metric values (with same keys as input dict metrics)
                dict
                    Loss values (with same keys as those initially passed to model.init).
                    Additionally, a total_loss key is added
                list
                    Arbitrary number of predictions as np.array

                """

        loss_vals = {}
        metric_vals = {}
        image_name_real_fl = "real_images_frontlight"
        image_name_real_bl = "real_images_backlight"
        image_name_fake_fl = "fake_images_B1_frontlight"
        image_name_fake_bl = "fake_images_B1_backlight"
        image_name_fake_fl_enc = "fake_images_frontlight_encoded"
        image_name_fake_bl_enc = "fake_images_backlight_encoded"
        mask = "segmentation mask"

        input_B = data_dict.pop('data')
        input_A = data_dict.pop('seg')

        inputs = np.concatenate((input_A, input_B), axis=1)

        real_fl = input_B[:, :3, :, :]
        real_bl = input_B[:, 3:, :, :]
        real_bl = np.concatenate([real_bl, real_bl, real_bl], 1)

        if model.training == True:
            z_rand = np.random.normal(size=(input_A.shape[0], model.latent_dim))
            z_recon_mu, z_encoded_mu, z_encoded_log_sigma, D_real, D_fake, D_fake_encoded, image_ab, image_ab_encoded, losses, *_ = model.run(
                inputs, z_rand)

            fake_fl = image_ab[:, :3, :, :]
            fake_bl = image_ab[:, 3:, :, :]
            fake_bl = np.concatenate([fake_bl, fake_bl, fake_bl], 1)
            fake_fl_enc = image_ab_encoded[:, :3, :, :]
            fake_bl_enc = image_ab_encoded[:, 3:, :, :]
            fake_bl_enc = np.concatenate([fake_bl_enc, fake_bl_enc, fake_bl_enc], 1)

            fake_fl = (fake_fl + 1) / 2
            logging.info({"images": {"images": fake_fl, "name": image_name_fake_fl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})

            fake_bl = (fake_bl + 1) / 2

            logging.info({"images": {"images": fake_bl, "name": image_name_fake_bl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})

            fake_fl_enc = (fake_fl_enc + 1) / 2
            logging.info({"images": {"images": fake_fl_enc, "name": image_name_fake_fl_enc,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})

            fake_bl_enc = (fake_bl_enc + 1) / 2

            logging.info({"images": {"images": fake_bl_enc, "name": image_name_fake_bl_enc,
                                     "title": "output_image", "env_appendix": "_%02d" % fold}})

        elif model.training == False:
            fake_fl = []
            fake_bl = []
        
            image_ab = []
            image_ab_encoded = []
            z_hist =[]
            image_name_fake_fl = "val_" + str(image_name_fake_fl)
            image_name_fake_bl = "val_" + str(image_name_fake_bl)
            image_name_fake_fl_enc = "val_" + str(image_name_fake_fl_enc)
            image_name_fake_bl_enc = "val_" + str(image_name_fake_bl_enc)

            for i in range(model.n_latent):
                z_rand = np.random.normal(size=(input_A.shape[0], model.latent_dim))
                z_lin = np.zeros(shape=(input_A.shape[0], model.latent_dim))
                z_lin[0][:] = (float(i)/model.n_latent - 0.5) * 2.0
                z_recon_mu, z_encoded_mu, z_encoded_log_sigma, D_real, D_fake, D_fake_encoded, image_ab_, image_ab_encoded_, losses = model.run(inputs, z_rand)
                z_hist.append(z_rand)

                fake_fl_ = image_ab_[:, :3, :, :]
                fake_fl_ = (fake_fl_ + 1) / 2
                fake_fl.append(fake_fl_[0])


                fake_fl_ = image_ab_encoded_[:, :3, :, :]
                fake_fl_ = (fake_fl_ + 1) / 2


                fake_bl_ = image_ab_[:, 3:, :, :]
                fake_bl_ = np.concatenate([fake_bl_, fake_bl_, fake_bl_], 1)
                fake_bl_ = (fake_bl_ + 1) / 2
                fake_bl.append(fake_bl_[0])



                fake_bl_ = image_ab_encoded_[:, 3:, :, :]
                fake_bl_ = np.concatenate([fake_bl_, fake_bl_, fake_bl_], 1)
                fake_bl_ = (fake_bl_ + 1) / 2

                image_ab.append(image_ab_)

            fake_fl = np.array(fake_fl)
            logging.info({'image_grid': {"image_array": fake_fl, "name": image_name_fake_fl,
                                     "title": "output_image", "env_appendix": "_%02d" % fold, "nrow": 1}})
            fake_bl = np.array(fake_bl)
            logging.info({'image_grid': {"image_array": fake_bl, "name": image_name_fake_bl,
                                 "title": "output_image", "env_appendix": "_%02d" % fold, "nrow": 1}})
            image_ab = np.array(image_ab)
            image_ab_encoded = np.array(image_ab_encoded)

            hist_name = "sampled_z"

            logging.info({"histogram": {"array": np.array(z_hist), "name": hist_name,
                                        "title": "output_image", "env_appendix": "_%02d" % fold}})




        for key, loss_val in losses.items():
            loss_vals[key] = loss_val

        if model.training == False:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key, metric_fn in metrics.items():
                metric_fl = []
                metric_bl = []
                for i in range(model.n_latent):
                    fl = fake_fl[i].reshape(1, *fake_fl[i].shape)
                    metric_fl.append(metric_fn(real_fl, fl))
                    bl = fake_bl[i].reshape(1, *fake_bl[i].shape)
                    metric_bl.append(metric_fn(real_bl, bl))

                metric_vals[key + '_FL_mean'] = np.mean(np.array(metric_fl))
                metric_vals[key + '_BL_mean'] = np.mean(np.array(metric_bl))


            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

            image_name_real_fl = "val_" + str(image_name_real_fl)
            image_name_real_bl = "val_" + str(image_name_real_bl)

            mask = "val_" + str(mask)


        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})
        gt = input_A

        logging.info({"images": {"images": np.concatenate([gt, gt, gt], 1), "name": mask,
                                 "title": "input_image", "env_appendix": "_%02d" % fold}})

        real_fl = (real_fl + 1) / 2

        logging.info({"images": {"images": real_fl, "name": image_name_real_fl,
                                 "title": "input_image", "env_appendix": "_%02d" % fold}})

        real_bl = (real_bl + 1) / 2

        logging.info({"images": {"images": real_bl, "name": image_name_real_bl,
                                 "title": "input_image", "env_appendix": "_%02d" % fold}})

        if model.training == True:
            return metric_vals, loss_vals, [D_real, D_fake, D_fake_encoded, image_ab, image_ab_encoded]

        else:

            return metric_vals, loss_vals, [image_ab, image_ab_encoded]

# !pip install tensorflow tensorflow-gpu matplotlib tensorflow-datasets ipywidgets

import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.models import Model

import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

# Import danych fashion_mnist
ds = tfds.load('fashion_mnist', split='train')

# Utworzenie iteratora po zbiorze danych oraz ich wyświetlenie
dataiterator = ds.as_numpy_iterator()



# Wyświetlanie przykładowych obrazów
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx in range(4):
#     sample = dataiterator.next()
#     ax[idx].imshow(np.squeeze(sample['image']))
#     ax[idx].title.set_text(sample['label'])

# plt.show()

# Funkcja służąca do normalizacji obrazów przez skalowanie
def scale_images(data):
    image = data['image']
    return image / 255

# Skalowanie obrazów
ds = ds.map(scale_images)
# Cachowanie zbioru danych do szybszego procesowania + podział na partie po 128
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
# Prefetch - każda partia zostaje wczytywana po połowie, aby nie obciążać zbytnio procesu
ds = ds.prefetch(64)

# Funkcja odpowiadająca za stworzenie generatora
def build_generator():
    model = Sequential()

    # Wektor ze 128 pozycjami 7x7
    model.add(Dense(7 * 7 * 128, input_dim=128))
    # Pyrzpisanie wagi dla wartości ujemnych w min,max
    model.add(LeakyReLU(0.2))
    # Zmiana na tensor trójwymiarowy - 7x7x128
    model.add(Reshape((7, 7, 128)))

    # Upsampling + warstwa konwolucyjna 5x5 #1
    # Wygląda to mniej więcej tak, że dla każdego 7x7x128 zostaje prztworzony tensor 7x7x1. Ostateczny rezultat to tensor 7x7x128, ponieważ zastosowane zostało tutaj 128 kerneli filtrujących
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling + warstwa konwolucyjna 5x5 #2 - obecnie obrazy mają wymiary 28x28
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # warstwa konwolucyjna 5x5 #3
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # warstwa konwolucyjna 5x5 #4
    model.add(Conv2D(128, 4, padding='same'))
    model.add(LeakyReLU(0.2))

    # Warstwa konwolucyjna do 1 kanału wraz z funkcją aktywacyjną, aby podkreślić największe wartości, a wyminąć te marginalne. Zwraca tensor 7x7x1 z wartościami od 0 do 1
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

    return model
# Stworzenie generatora
generator = build_generator()
# generator.summary()


# Wygenerowanie obrazów z wyżej utworzonego generatora
img = generator.predict(np.random.randn(4,128,1))
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(img):
    ax[idx].imshow(np.squeeze(img))
    ax[idx].title.set_text(idx)

# print(img.shape)
# plt.show()

# Tworzenie dyskryminatora

def build_discriminator():
    model = Sequential()

    # Warstwy konwolucyjne przyjmują wartość filtrów x^i, rozpoczynając od 32, zgodnie z zasadą od ogółu (takie jak krawędzie lub obszar), do szczegółu (np. wyłapania guzika na koszuli lub podobnych mniejszych elementów)

    # warstwa konwolucyjna #1
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    # Lekki przeciek wartości ujemnych. Zamiast relu zwracającego wartość 0 dla wartości x <= 0, to zwraca a x x dla wartości x <= 0
    model.add(LeakyReLU(0.2))
    # Warstwa dropout losowo wyłącza 40% neuronów podczas treningu. Ma to na celu zapobieganie występieniu "overfitting"
    model.add(Dropout(0.4))

    # warstwa konwolucyjna #2
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # warstwa konwolucyjna #3
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # warstwa konwolucyjna #4
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Spłaszczenie tensora (jeżeli dobrze liczę to o wymiarach 12x12x1) do jednowymiarowego wektora o długości 144
    model.add(Flatten())
    model.add(Dropout(0.4))
    #
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()

# Tworzenie optymalizatorów dla generatora i dyskryminatora oraz funkcji straty
# Zastosowanie Algorytmu ADAM do dostosowywania parametrów w modelach
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
# Zastosowanie BinaryCrossentropy (−(y⋅log(p)+(1−y)⋅log(1−p)) gdzie y to rzeczywista etykieta (0 lub 1), p to przewidywana etykieta
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()

# Definicja klasy FashionGAN jako Modelu - dziedziczone z super klasy są wszystkie argumenty liczbowe i tekstowe. Dodatkowo zostają podłączone generator i dyskryminator
class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

# Przygotwanie do procesu trenowania
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)

        # Tworzenie atrybutów dla funkcji straty i optymalizatorów
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        # Pobieranie danych
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Trenowanie dyskryminatora
        with tf.GradientTape() as d_type:
            # Przekazywanie prawdziwych i fałszywych obrazów do modelu dyskryminatora
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            # Zebranie wyników
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Tworzenie etykiet dla prawdziwych i fałszywych obrazów
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Dodanie szumu do prawdziwych wyjść
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Obliczanie straty
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Wsteczna propagacja wag przez obiekt optymalizatora dla dyskryminatora
        # zip() łączy gradienty dgrad z odpowiednimi zmiennymi podlegającymi treningowi
        dgrad = d_type.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Trenowanie generatora
        with tf.GradientTape() as g_type:
            # Generowanie nowych obrazów
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Przewidywanie etykiet - czym więcej 1 tym lepiej dla realizacji celu generatora, jakim jest oszustwo dyskryminatora
            predicted_labels = self.discriminator(gen_images, training=False)

            # Obliczanie straty generatora
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Wsteczna propagacja dla generatora
        ggrad = g_type.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}



class ModelMonitor(Callback):
    # num_img - oznacza liczbę obrazów do wygenerowania; latent_dim - długość wektora wejściowego
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
    # Tworzenie pustego tensora dla 3 zdjęć, gdzie wektor ma wielkość danych wejściowych;
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))



fashgan = FashionGAN(generator, discriminator)
fashgan.compile(g_opt, d_opt, g_loss, d_loss)

# Train model

hist = fashgan.fit(ds, epochs=20, callbacks=[ModelMonitor()])

# Review Performance

plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()

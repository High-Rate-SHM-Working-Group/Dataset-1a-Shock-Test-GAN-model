GENERATOR_NAME = 'conditional_af_accel_generator_v3'
LATENT_DIM = 128
MODEL_OUTPUT_LENGTH = 5_000
LABEL_LENGTH = 4
LABEL_CHOICES = ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                 [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
                 [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1],
                 [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                 [1, 1, 1, 1])
# should be 15 because all except empty set is 2**4 - 1
NUM_LABEL_CHOICES = 15  # len(LABEL_CHOICES)
SINGLE_LABELS = LABEL_CHOICES[:4]
MULTI_LABELS = LABEL_CHOICES[4:]
SCALARS = [21.41246811, 23.22566933, 25.41179673, 26.63817594, 40.94323122,
           51.24551213, 61.49926574, 69.78805778, 22.03701639, 27.81502613,
           29.61584453, 30.1159875, 40.96605813, 51.16870271, 61.32964213,
           71.56054567]

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

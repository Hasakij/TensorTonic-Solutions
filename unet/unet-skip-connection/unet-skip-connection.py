import numpy as np

def crop_and_concat(encoder_features: np.ndarray,
                    decoder_features: np.ndarray) -> np.ndarray:
    """
    Returns the centered encoder crop concatenated with decoder features.
    """
    h_enc, w_enc = encoder_features.shape[1], encoder_features.shape[2]
    h_dec, w_dec = decoder_features.shape[1], decoder_features.shape[2]

    # Compute starting offsets for center crop
    h_offset = (h_enc - h_dec) // 2
    w_offset = (w_enc - w_dec) // 2

    # Slice the encoder features to match decoder size
    encoder_cropped = encoder_features[:, h_offset:h_offset+h_dec, w_offset:w_offset+w_dec, :]

    # Concatenate along the channel axis
    merged = np.concatenate([encoder_cropped, decoder_features], axis=-1)
    return merged.astype(np.float64)
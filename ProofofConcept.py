# %%
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
# %%

model = SentenceTransformer('sentence-transformers/sentence-t5-large')
base_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
print(model)
# %%

# Example sentences

sentence = ["This is a sentence", "These are sentences", "Milk is white",
            "Architecture is the art and technique of designing  buildings",
            "Please don't skip this one minute read", "woman life freedom",
            "Architecture is the art and technique of designing  buildings",
            "Architecture is not only art but also engineering"]

embeddings1 = model.encode(sentence)
embeddings2 = base_model.encode(sentence)

print(embeddings1.shape)


# %%
def norms(tnsr: np.ndarray) -> np.ndarray:
    """
    This function makes norm matrix (euclidean) for all differences of the
    given sentences.

    Args:
        np.ndarray: output of the T5 sentence encoder vector

    Returns:
        np.ndarray: A symmetrical matrix containing the
        differences of all sentence vectors
    """
    mtx = np.tile(tnsr, (tnsr.shape[0], 1, 1))
    return np.linalg.norm(mtx - np.transpose(mtx, [1, 0, 2]), axis=-1)
# %%


norms1 = norms(embeddings1)

fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(norms1, vmin=0, vmax=0.9)
plt.colorbar()
for i in range(norms1.shape[0]):
    for j in range(norms1.shape[0]):
        text = ax.text(j, i, round(norms1[i, j], 2),
                       ha="center", va="center", color="k")

plt.show()


norms2 = norms(embeddings2)

fig, ax = plt.subplots(figsize=(10, 10))
plt.imshow(norms2, vmin=0, vmax=0.9)
plt.colorbar()
for i in range(norms2.shape[0]):
    for j in range(norms2.shape[0]):
        text = ax.text(j, i, round(norms2[i, j], 2),
                       ha="center", va="center", color="k")

plt.show()

import streamlit as st
import numpy as np
import torch
import torchvision
import faiss
from PIL import Image
from torchvision import transforms
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Chemin vers le dossier de cache PyTorch
cache_dir = os.path.expanduser("~/.cache/torch/hub")

# Supprimer le cache si le dossier existe
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cache supprimé avec succès : {cache_dir}")
else:
    print("Aucun cache PyTorch trouvé.")

st.set_page_config(layout="wide")

# Charger l'index avec Faiss
@st.cache_data
def load_index():
    try:
        all_vecs = np.load("./resnet_index/all_vecs.npy").astype('float32')
        all_names = np.load("./resnet_index/all_names.npy")
        index = faiss.IndexFlatL2(all_vecs.shape[1])
        index.add(all_vecs)
        return index, all_names
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'index : {e}")
        return None, None

index, names = load_index()

if index is None or names is None:
    st.stop()

st.title("Recherche d'images similaires")
guploader = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])
st.write("___")

if guploader:
    col1, col2 = st.columns([2,5])
    with col1:
        try:
            image = Image.open(guploader).convert("RGB")
            st.image(image, caption="Image téléchargée")
        
            # Prétraitement de l'image
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0)
        
            # Chargement du modèle ResNet-18 local
            model_path = "C:\\Users\\HP\\Documents\\dockerFom\\machinelearning\\New folder\\resnet_models\\hub\\checkpoints\\resnet18-f37072fd.pth"
            model = torchvision.models.resnet18()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
        
            # Extraction des caractéristiques
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
        
            model.avgpool.register_forward_hook(get_activation("avgpool"))
        
            with torch.no_grad():
                _ = model(img_tensor)
                input_vec = activation["avgpool"].numpy().squeeze().astype('float32')
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")
            st.stop()
    
    with col2:  
        contenair_1 = st.container()
        with contenair_1:
            try:
                # Recherche des images similaires avec Faiss
                st.write("Recherche en cours...")
                progress_bar = st.progress(0)
                distances, indices = index.search(np.expand_dims(input_vec, axis=0), 10)  # Passer à 10 images similaires
                progress_bar.progress(100)

                st.write("Images similaires :")
                cols = st.columns(5)
                for i in range(0, 10, 5):  # Afficher sur deux lignes
                    row_cols = st.columns(5)
                    for col, idx in zip(row_cols, indices[0][i:i+5]):
                        img_path = os.path.join("./datasets", names[idx])
                        col.image(Image.open(img_path))
            except Exception as e:
                st.error(f"Erreur lors de la recherche d'images similaires : {e}")
else:
    st.warning("Veuillez télécharger une image pour lancer la recherche.")
import streamlit as st
from model2 import check_jpg, convert_image, predict_image

# ----------- General things
st.title('Semantic Segmentation')
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Benchmarking"])

st.sidebar.markdown("""---""")
st.sidebar.write("Github [Repo](https://github.com/WhiteWolf47/cscapes_semantic_segmentation)")
st.sidebar.image("model.png", width=100)

if page == "Predictor":
    # ----------- Inputs
    st.markdown("Upload image for segmentation")
    upload_columns = st.columns([1, 1])

    # File upload
    file_upload = upload_columns[0].expander(label="Upload a jpg file")
    uploaded_file = file_upload.file_uploader("Choose jpg file", type=['jpg'])

    # Smiles input
    '''smiles_select = upload_columns[0].expander(label="Specify SMILES string")
    smiles_string = smiles_select.text_input('Enter a valid SMILES string.')'''

    # If both are selected, give the option to swap between them
    '''if uploaded_file and smiles_string:
        selection = upload_columns[1].radio("Select input option", ["File", "SMILES"])'''

    
    
    if uploaded_file:
        # Save it as temp file
        temp_filename = "temp.jpg"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        #loaded_molecule = check_jpg(temp_filename)
        #load the image
        image = tf.io.read_file(temp_filename)

    # ----------- Validation
    st.info("This image appears to be valid :ballot_box_with_check:")
    #pil_img = draw_molecule(loaded_molecule)
    upload_columns[1].image(image, width=200)
    submit = upload_columns[1].button("Get predictions")

    # ----------- Submission
    st.markdown("""---""")
    if submit:
        with st.spinner(text="Fetching model prediction..."):
            # Convert image to 512x256
            input_img = convert_image(image)
            # Call model endpoint
            prediction = get_model_predictions(graph)

        # ----------- Ouputs
        st.markdown("Model predictions")
        output_columns = st.columns([1, 1])
        output_columns[0].image(input_img, width=200)
        output_columns[1].image(prediction, width=200)

else:
    st.markdown("This page is not implemented yet :no_entry_sign:")
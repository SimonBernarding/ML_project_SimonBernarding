import streamlit as st
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Import of chart packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

# Set page config
st.set_page_config(page_title="Flight Delay Presentation", page_icon="images/flight.ico", layout="wide")

# Load and encode the image
image = Image.open("images/flight.png")
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Create inline HTML for image and text
inline_title = f"""
<div style="display: flex; align-items: center; white-space: nowrap;">
    <span style="font-size: 2em; font-weight: bold;">Flight Delay</span>
    <img src="data:image/png;base64,{img_str}" style="height: 3em; margin-left: 10px;">
</div>
"""

# Display the inline title
st.write(inline_title, unsafe_allow_html=True)

# Sidebar for navigation
page = st.sidebar.radio("Overview", ["Case Study", "Data", "Analysis", "Model"])
page = st.sidebar.radio("Overview", ["Case Study", "Data", "Analysis", "Model"])

# Introduction page
if page == "Case Study":
    st.header("Case Study")
    st.write("Flight delays are a significant concern in the aviation industry, affecting passenger satisfaction, airline operations, and overall transportation efficiency. This dataset focuses on flight delays experienced by Tunisian airlines, providing valuable insights into the patterns and causes of delays in the North African aviation sector.")
    
    st.subheader("Features")
    st.write("- Departure & Destination")
    st.write("- Time data")
    st.write("- Flight ID & Plane ID")

    st.subheader("Target")
    st.write("- Delay")



# Data Analysis page
elif page == "Data":
    st.header("Data")

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_csv("data/Train.csv")

    df = load_data()

    st.subheader("DataFrame")
    st.write(df.head())

    st.write("- In total 107833 entries")
    st.write("- No missing values")
    st.write("- No duplicates")

    def load_data():
        return pd.read_csv("data/data.csv")

    df = load_data()

    # Create the scatter plot
    fig = px.scatter_mapbox(df, lat="latitude_arr", lon="longitude_arr", 
                            color_discrete_sequence=["teal"], zoom=2)

    # Custom mapbox style with opacity
    mapbox_style = {
        "version": 8,
        "sources": {
            "osm": {
                "type": "raster",
                "tiles": ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
                "tileSize": 256,
                "attribution": "&copy; OpenStreetMap Contributors",
                "maxzoom": 19
            }
        },
        "layers": [{
            "id": "osm",
            "type": "raster",
            "source": "osm",
            "paint": {"raster-opacity": 0.8}  # Set the opacity here
        }]
    }

    # Update layout with custom mapbox style
    fig.update_layout(mapbox_style=mapbox_style)

    # Update layout for size and title
    fig.update_layout(
        height=600,  # Adjust height as needed
        title={
            'text': "Distribution of Airports",
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Add a data table below the map
    # if st.checkbox("Show raw data"):
        # st.write(df)


# Other pages
elif page == "Analysis":
    st.header("Analysis")
    # Add your analysis content here
    
    st.write("- about 64 % of flights are delayed")
    st.write("- mean: 75 min, std: 139 min, median: 30 min, max: 3451 min")

    image = Image.open('images/target_distribution_delayed.png')
    st.image(image, caption='Target distribution (delay time in min)', use_column_width=False, width=800)

    st.write("- 3 most frequent flight routes: ORY-TUN, TUN-ORY, TUN-TUN")
    st.write("- departure airport same as arrival airport: could be e.g. flight school")
    st.write("- top 3 flight routes with most delay time: ORY-TUN, TUN-ORY, IST-TUN")
    #st.write("- ORY-TUN (Paris-Tunis) contributes ~5 % to total delay time")
    st.write("\n\n")
    st.write("\n\n")
    st.write("Get additional information:")

    st.write("- Airline can be extracted out of flight number")
    st.write("- Airplane model and manufacturer can be extracted out of aircraft number")
    st.write("\n\n")
    st.write("\n\n")

    st.subheader("Feature engineering")
    st.write("- binary encoding for categorical features e.g. airport, airline, producer")
    st.write("- drop unnecessary columns")
    st.write("- convert target into categories")
    st.write("- standardization of data before splitting and training")

    st.write("\n\n")
    st.write("\n\n")

    #data = {'Interval': ["No delay", "0 - 30 min", "30 - 60 min", "60 - 120 min", "120 - 240 min", "> 240 min"]}
    #df_cat = pd.DataFrame(data)
    #df_cat.index += 1
    #st.dataframe(df_cat)

    st.write("\n\n")
    st.write("\n\n")


    image = Image.open('images/delay_categories_distribution.png')
    st.image(image, use_column_width=True)

    

elif page == "Model":
    st.header("Model")
    st.subheader("Used Models")
    lst_models = [
        "0 Logistic Regression (BM)",
        "1 SGD",
        "2 KNeighbors",
        "3 Decision Tree",
        "4 Random Forest",
        "5 XGB",
        "6 Ada Boost",
        "7 Bagging",
        "8 Extra Trees",
        "9 Gradient Boosting",
        "10 Stacking (1-9)",
        "11 Max Voting (1-9)"
]
    for model in lst_models:
        st.write("-", model)
        
    st.subheader("Score")
    st.write("Precision = What proportion of the flights predicted delayed are actually delayed?")
    st.write("Recall = What proportion of the delayed flights was predicted correctly?")
    st.write("F1-Score")
    
    ####### plot model f1 score ######### START ############
    
    # import data
    df_f1_sc = pd.read_csv("data/df_f1_sc.csv")
    
    #st.dataframe(df_f1_sc)
    
    # Farbenliste erstellen
    colors = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B',
    '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', '#AEC7E8', '#FFBB78'
    ]
   
    # Säulendiagramm erstellen
    fig, ax = plt.subplots(figsize=(8, 4))

    bar_width = 0.35
    index = range(len(df_f1_sc))

    # Bars für f1_train und f1_test mit unterschiedlichen Farben je Modell
    for i, (train, test) in enumerate(zip(df_f1_sc['f1_train'], df_f1_sc['f1_test'])):
        ax.bar(i, train, bar_width, label=f'f1_train' if i == 0 else "", color=colors[i], alpha=0.7)
        ax.bar(i + bar_width, test, bar_width, label=f'f1_test' if i == 0 else "", color=colors[i], alpha=0.4)

    # Achsenbeschriftungen
    ax.set_xlabel('model')
    ax.set_ylabel('F1 Score (weighted average)  [-]')
    ax.set_title('F1 Train and Test Scores for each model')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(df_f1_sc.model)

    # Legende hinzufügen
    ax.legend()

    # Diagramm anzeigen
    #plt.tight_layout()
    #plt.show()
    
    # Display the plot in Streamlit
    plt.xticks(rotation=90)
    st.pyplot(fig, use_container_width=True)
    
        ####### plot model f1 score ######### END ############
    
        ####### plot f1 score category-wise ####### START ########
    
    image = Image.open('images/F1_Train_Test_all_models_percategory.png')
    st.image(image, caption='F1 TEST score per category', use_column_width=False, width=600)    
      ####### plot f1 score category-wise ####### END ########  
        
              ####### plot f1 score category-wise ####### END ########
            
    st.subheader("Extra Features")
    st.write("- historical weather data: threshold conditions when not to fly (rain, storms, ...)")
    st.write("- wind direction: difference for flights with / against direction of wind?")
    st.write("- events prohibiting flight departure (major political events, economy crisis, etc.) ")

    st.subheader("Model optimization")
    st.write("- hyper parameter optimization: grid search cv or random grid search cv does sometimes lead to worse models. Why?")
    st.write("- SMOTE for inbalanced data?")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Done by [Alexander Kopp](https://github.com/KoppAlexander), [Simon Bernarding](https://www.linkedin.com/in/simon-bernarding/) & [Michel Penke](https://michelpenke.de/portfolio/)")
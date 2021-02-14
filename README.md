# Disaster Response Pipeline Project
This project tackles the disaster response dataset provided by Figure Eight.
Once a disaster strikes anywhere, people usually send out messages whether over social media or to responder groups. The amount of messages received can be overwhelming if it were to be processed manually. Therefore, it is very important to label those messages automatically in order to direct them to the relevant group.

### Project Structure
   - app: Contains the Flask application which will serve the web pages. The main page shows multiple charts exploring the data, and a field where you can enter a message and see its classification.
   - data: Contains all data and data processing files:
        - The raw data CSV files
        - The ETL file which reads the data, cleans it and then stores it into the corresponding database
        - The databasse file into which the dataset will be loaded
   - models: Contains the ML pipeline which will: 
        - Read the data from the database
        - Preprocess the data. Given that it is textual data, it will convert it to lower case, remove special characters, tokenize, remove stop words, stem and lemmatize
        - Build a machine learning pipeline for multi-class classification relying on random forest at its heart. Also, it utilizes grid search to tune parameters to maximize f1-score
        - Finally it outputs the model performance and saves the best model as a pickle file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

###### Please note: If training the model (running the ml pipeline) is taking a very long time, please consider commenting out one of the parameters options for grid search. You can also, disable the use of grid search if needed.
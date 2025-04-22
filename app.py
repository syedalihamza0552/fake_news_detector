import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_model_and_vectorizer():
    """
    Load the trained model and TF-IDF vectorizer from ./dataset/.
    
    Returns:
        model: Trained scikit-learn model
        vectorizer: Fitted TfidfVectorizer
    """
    with open('./dataset/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./dataset/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def preprocess_text(text):
    """
    Preprocess input text to match training pipeline.
    
    Args:
        text (str): Input text to preprocess
    
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, HTML tags, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens
    return ' '.join(tokens)

def compute_features(text, vectorizer):
    """
    Convert text to TF-IDF features and add text length.
    
    Args:
        text (str): Preprocessed text
        vectorizer: Fitted TfidfVectorizer
    
    Returns:
        scipy.sparse.csr_matrix: Feature matrix
    """
    # Compute TF-IDF features
    tfidf_features = vectorizer.transform([text])
    
    # Compute text length
    text_length = np.array([[len(text.split())]])
    
    # Combine features
    features = hstack([tfidf_features, text_length])
    
    return features

def main():
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    # Streamlit app
    st.title("Fake News Detection")
    st.write("Enter a news article to check if it's Fake or True.")
    
    # Text input
    user_input = st.text_area("News Article Text", height=200)
    
    if st.button("Predict"):
        if user_input.strip():
            # Preprocess input
            clean_text = preprocess_text(user_input)
            
            # Compute features
            features = compute_features(clean_text, vectorizer)
            
            # Predict
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0][prediction] * 100
            
            # Display result
            label = "True" if prediction == 1 else "Fake"
            st.success(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
            
            # Disclaimer
            st.warning("Note: This model is not 100% accurate and may misclassify some articles. Use results as a guide, not a definitive judgment.")
        else:
            st.error("Please enter some text to analyze.")
    
    # Example inputs
    st.subheader("Try Example Articles")
    if st.button("Example Fake News"):
        st.text_area("News Article Text", value="patrick henningsen 21st century wirethere exists famous quote often attributed 1918 u senator hiram warren johnson said first casualty war come truth could said waco ruby ridge federal government version recent event burn oregon really happened highway 395 sure yet meantime seems mainstream medium already decided guilty notregarding tuesday evening event oregon standoff federal agent shot killed rancher robert lavoy finicum campaign disinformation appears rapidly underway following prime exhibit mainstream medium spin story favor government version eventsthe washington post brash headline read report lavoy finicum armed handgun reached waistband shot absence actual evidence even analysis support deceptive headline instead washington post writer mike miller attempt weave together classic example yellow journalismany speculation public skepticism could easily put rest fbi releasing video footage event question screen shot washington postmiller begin piece establishing finicum guilt murder nature fact robert lavoy finicum seen previously carrying gun everyone knew lavoy finicum kept colt 45 hip common knowledge could help explain occupier oregon died miller article hammer home federal thesis injured dead protester brought occupier ample opportunity leave peacefully fbi special agent greg bretzing said wednesday morning brought word according fbi help medium dead rancher lavoy finicum guilty court public opinionmarginalizing witnessmiller immediate move cast doubt testimony 18 yr old victoria sharp passenger truck finicum driving crassly describing young woman selfdescribed witness add another selfdescribed witness dispute saying authority opened fire getaway chase finicum hand shot stoic ammon bundy ryan bundy lavoy finicum wildlife refuge last weekinstead miller relies heavily youtube video recorded uploaded man named mark mcconnell scene actual shooting admits one mile away federal agent reportedly killed finicum admittedly mcconnell testimony claiming finicum charged officer secondhand possibly thirdhand depending accuracy attempting relaying passenger riding vehicle sharp time ryan payne shawna coxhere mcconnell effect issuing proxy statement behalf victoria sharp fellow passenger shawn cox ryan payne hardly admissible cox payne currently federal custody able verify accuracy mcconnell reiteration yet mainstream medium outlet proppingup mcconnell account somehow factualclearly nudging exercise carefully attempting minimize discredit young female witness sharp post miller make blatantly obvious subtly insert sharp said 18 year old car finicum said 18 yr old context mcconnell account conflict statement made victoria sharp said 18yearold car finicum one would think washington post headline journalist would ample resource confirm whether sharp truck notwhy mainstream medium outlet quick try discredit testimony victoria sharp unlike mainstream medium goto man mcconnell victoria sharp actual eye witness saw firsthand fateful event question described detail immediately event phone interview amazing newspaper like washington post presumably good reputation high journalistic standard one would think anyway would grab recycled secondhand account first appeared tabloid news website raw storywhat post attempting clear try align secondhand speculative guess mcconnell vague anonymous statement leaked fbi cnnsadly news consumer uncommon network like cnn often accept vague anonymous leak source order construct official narrative event absolutely zero accountability either government source news networkit almost miller writing piece behalf fbiat least washington post playing supporting role acting additional pr channel echo chamber claim neither cnn government required either confirm deny source say finicum reaching waistbandrather question claim fashioned cnn source journalist mike miller opposite try reinforce government narrativecnn international report hinge anonymous government source claiming finicum killed federal agent reaching towards waistband miller writes moment shot authority tuesday afternoon finicum led highspeed getaway attempt reached waistband prompting authority open fire according cnn reportthe report cite anonymous law enforcement official confirmed washington post corroborated statement occupier said traveling finicum time traffic stop fact medium chosen label incident traffic stop indicates obvious agenda designed minimize obvious preplanned elaborate ambush fbi oregon state policethe washington post try enhance cnn phantom report aligning government narrative secondhand story presented mark mcconnell proppingup government version eventsthis unnamed source cited cnn also told network incident captured video fbi authority decided whether release video one indication anonymous leak cnn likely bogus fact statement say finicum reached waistband chose say reached gun offer government convenient exit previous statement unincriminating video ever surface public viewing addition fact alleged damning video already released could mean video footage question support government innuendo story leaked via governmentmedia information channelsany public distrust skepticism federal authority would immediately dispelled speed release video footage dozen agent surveillance one video feed taken including body gun sight camera event whether fast happen remains seentwo version eventswhat abundantly clear listening mcconnell sharp conflicting testimony mcconnell statement quite obviously unreliable full selfcontradictions caveat sharp clear concisethe following testimony victoria sharp recorded immediate aftermath shooting oregon highway 395 anyone looking story see clearly protester vehicle traveling prearranged community event nearby small town john day event supported neighboring grant county sheriff glenn palmer palmer sympathetic bundy protest publicly called release hammond family belief wrongfully imprisoned oregon live reported palmer fbi roadblock highway 395 alongside large contingent armed federal agent true would indicates palmer prior knowledge federal ambush importantly clearly ceded legal consitutional authority county chief law enforcement official fbi special agent charge greg bretzing absent public statement palmer matter onlooker speculate situationimportant note update 13016 according recent report pedro quintana central oregon local affiliate news channel 21 sheriff palmer federal roadblock prior knowledge fbi ambush bundy convoy according chat session relayed pedro quintana report palmer asked online chat witness lavoy shooting palmer reply shooting hear close 4 heard 545pm news 21continued palmer responded knowledge anything plan coming public meeting went say fbi oregon state police know shared nothing know allowed earlier mainstream medium report palmer present fbi roadblock would lead bundy supporter believe palmer betrayed bundys lavoy finicum one main source main source original sheriff palmer controversy appears none oregon live staff writer le zaitz slippedin key piece line report say said know long roadblock would place grant county sheriff glenn palmer surprisingly setoff virtual backlash online grant county sheriff evidenced following screen shot google search link found case medium dirty trick oregon live certainly oregon live coverage burn event skewed beginning running character assassination piece malheur occupier effort discredit protest sheriff palmer clearly record sympathetic hammond ranch protest recently opened constructive dialogue bundys occupier malheur wildlife refuge pedro quintana news 21 report indeed accurate one might conclude oregon live put piece disinformation likely designed divide protest movement create dissension militia activist rank however sheriff glenn palmer come issue clear statement declaring people continue speculate really happened end updatenote see depth legal analysis sheriff palmer fbi operation free capitalistamazingly post writer mike miller go admit premeditated ambush protester vehicle highway 395 yet call ambush according cnn report fbi oregon state police watching occupier come go malheur national wildlife refuge near burn day spotted rare opportunity nab militant movement leader finicum ammon bundy ryan bundy handful others climbed two truck drove refuge meeting john day ore two hour north authority chose cold deserted stretch highway attempt traffic stop one vehicle complied order pulled allegedly driven finicum sped high speed according report noted although never washington post cnn protest leader ammon bundy met fbi negotiator morning continuing open dialogue going week plus fact sheriff palmer invited bundy fellow protester community event john day demonstrated federal government local sheriff acted bad faith drawing main protester town john day along preplanned route staging ambush overwhelming force hour later fact man dead definitely underline seriousness course action appears case deception part authoritiessimilarly right event mainstream medium calling event shootout even though obvious shot fired vehicle passenger granted deceptive language medium nothing new still indication deliberate attempt skew narrative favor law enforcement official version eventsupdate 1282016 1130pm et due public pressure intense speculation event tuesday evening youtube recording 18 yr old eye witness victoria sharp fbi released unedited aerial video footage tuesday evening incident took place along highway 395 according official fbi statement feel necessary show whole thing unedited interest transparency fbi video entitled complete unedited video joint fbi osp operation 01262016 show victim lavoy finicum exiting truck awkwardly least two foot snow clearly charge towards swat team hand clearly held high head exit truck confronted swat team shot multiple time marksman falling snow also scale size operation evident footage apprehension protester result mere traffic stop wrongly characterized multiple mainstream medium report fbi drone footage would appear support previous government medium claim cnn washington post others victim indeed reaching towards waistband gave federal state police justification unleash deadly force however still completely clear aerial footage whether finicum holstered said normally carry gun right hip fbi claiming 9mm gun inside left breast pocket also whether lowered arm shot multiple time example finicum hand air shot abdomen first man natural reaction would lower hand clutch wound additionally shot fired even vicinity confusion could ensued might prompted finicum reach weapon either way impossible make forensic determination without corresponding audio track would help determine shot fired finicum hand could seen lowering warning following image depict violence death reader may find disturbing watch finicum appears temporarily lose balance snow moment look hit agentswhat clear however video armed agent shoot disable suspect shot kill even laser sight trained many minute downed clearly multiple swat shooter could seen emerging wood killing finicum police left victim bleed death laying snow check least another 8 minutesflashbang round seen around 1400 mark fbi unedited version fbi said fired c gas non lethal round possibly pepper spray round rubber bullet said fired truck passenger inside begs question nonlethal round already play multiple swat shooter use many deadly round finicum exited truck hand upsee 21wire detailed report federal ambush highway 395 oregon hereread oregon standoff news 21st century wire oregon file", key="fake_example")
    if st.button("Example True News"):
        st.text_area("News Article Text", value="washington reuters president donald trump nominee u trade representative said tuesday belief china substantially manipulated currency past gain trade advantage unclear beijing still past judgment china substantial currency manipulator think weve lost lot job united state robert lighthizer told senator confirmation hearing whether china manipulating currency right weaken thats another question lighthizer said adding u treasury secretary steven mnuchin would person make determination trump pledged campaign declare china currency manipulator first day administration move would trigger demand bilateral negotiation issue done china past year spent hundred billion dollar foreign exchange reserve prop value yuan face capital outflow pressure", key="true_example")

if __name__ == "__main__":
    main()
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from pickle import load

selected = option_menu(
    menu_title = None,
    options=["Home", "Dataset", "Prediction"],
    icons = ['house', 'clipboard2-data', 'emoji-dizzy'],
    menu_icon = 'cast',
    default_index=0,
    orientation='horizontal'
)



if selected == 'Home':
    st.title('Diamond 💎')
    st.markdown('SHINE BRIGHT LIKE A DIAMOND')

    st.markdown("""---""")
    st.markdown("""---""")

    st.header('What Is a Diamond?')
    st.markdown("Deriving its name from the Greek word adámas (meaning 'unbreakable'), a diamond is a gemstone that is composed of carbon atoms in a crystal lattice arrangement. Diamonds can form naturally or be grown in a laboratory.")

    st.markdown("""---""")


    st.image('img/cut.jpg', caption='cut')
    st.markdown('Cut is the only diamond component not influenced by nature, and Mills considers this the most important of the 4Cs. This factor refers to the quality of the diamond"s cut, not the shape or size (although these can be interchangeable), and how well the stone is faceted, proportioned, and polished. This also determines how the diamond interacts with light. Brilliance, which is the diamond"s ability to return light to the eye, is measured solely by the stone"s cut (color and clarity have no impact). For any diamond shape, visually, cut is the first C to consider, followed by color, and, least as important, clarity (as long as the diamond has no visible imperfections).Per the GIA system, diamond cuts are graded as Excellent, Very Good, Good, Fair, and Poor. Cut grade doesn"t influence the cost as much as the other Cs, so Mills suggests always sticking within the Excellent to Very Good range for a well-cut stone that works best with light.')

    st.markdown("""---""")

    st.image('img/color.jpg', caption='color')
    st.markdown("Diamond colors fall under a D-Z scale, with D meaning completely colorless (and the most expensive), and Z having a light yellow hue. According to Mills, standard diamond quality falls within the D-J color grade. The shape of the diamond also influences its spot on the color scale. A round brilliant diamond, for example, hides color incredibly well, meaning you can go further down the scale without seeing any yellowing. However, longer diamond shapes, like oval and radiant, reveal color much easier. Keep in mind, though, diamond color is essentially personal preference and doesn't indicate quality whatsoever.")


    st.markdown("""---""")

    st.image('img/clarity.jpg', caption='clarity')
    st.markdown("This C involves the number of natural imperfections, called inclusions, present in the diamond, and whether you can see them with the unaided eye. The GIA grading scale rates diamonds from Flawless (FL) to Included (I). However, a stone doesn't have to be at the very top of the scale—Flawless or Very Very Slightly Included (VVS)—to look perfect and inclusion-free. It's all about how eye-clean the diamond appears, and Mills says this is what usually surprises people most when viewing diamonds in person. In fact, if an SI1 (Slightly Included) clarity diamond appears perfectly eye-clean, there is no visible difference between a VVS1 (Very Very Slightly Included) clarity stone of the exact same carat, color, and cut—minus about tens of thousands of dollars.")
    st.markdown("'There is no reason, in my opinion, to go higher than VS1 [Very Slightly Included] clarity for any diamond shape except emerald or Asscher,' says Mills. 'For all other shapes, starting at SI1 [Slightly Included] clarity and up, you should not normally see any imperfections visible to the naked eye. Sometimes even SI2 diamonds can be very eye-clean, as well, but generally stick with SI1 and up.'")


    st.markdown("""---""")

    st.image('img/carat.jpg', caption='carat')
    st.markdown("Last but not least, carat refers to a measurement of the actual weight of the diamond. According to GIA, one carat converts to 0.2 grams, which is essentially the same weight as a paper clip. Naturally, the larger the carat, the more expensive the diamond. Because no two diamonds are completely identical, carat should be viewed as a guideline, since it only determines the weight of the stone as opposed to the actual size.")

if selected == 'Dataset':
    df=pd.read_csv('data/diamonds.csv')

    st.title('Diamond Dataset💎')

    st.subheader('Shape of Datasets')
    st.dataframe(df.shape)

    df1 = df.drop('price',axis = 1)
    st.dataframe(df1)

if selected == 'Prediction':
    df=pd.read_csv('data/diamonds.csv')

    or_enc=load(open('models/ordinal_encoder.pkl','rb'))
    scaler = load(open('models/Standard_scaler.pkl', 'rb'))
    dt_regressor=load(open('models/dt_regressor.pkl','rb'))


    st.title('💎 Diamond Price Prediction 💎')


    with st.form('my_form'):
        carat=st.select_slider('Carat', options=df.carat.unique())
        cut=st.selectbox(label='Cut of Diamond', options=df.cut.unique())
        color=st.selectbox(label='Color of Diamond', options=df.color.unique())
        clarity=st.selectbox(label='Clarity level of Diamond', options=df.clarity.unique())
        depth=st.select_slider('Depth of Diamond', options=df.depth.unique())
        table=st.select_slider('Table of Diamond', options=df.table.unique())
        x = st.select_slider('Select Length of diamond in mm', options=df.x.unique())
        y = st.select_slider('Width of diamond in mm', options=df.y.unique())
        z = st.select_slider('Depth of diamond in mm', options=df.z.unique())


        btn = st.form_submit_button(label='Predict')

    if btn:
        if carat and cut and color and clarity and depth and table and x and y and z:
            query_num = pd.DataFrame({'carat':[carat], 'depth':[depth],'table':[table],'x':[x],'y':[y],'z':[z]})
            query_cat = pd.DataFrame({'cut':[cut], 'color':[color], 'clarity':[clarity]})   

            query_cat = or_enc.transform(query_cat)
            query_num = scaler.transform(query_num)

            query_point = pd.concat([pd.DataFrame(query_num), pd.DataFrame(query_cat)], axis=1)
            price = dt_regressor.predict(query_point)

            st.success(f"The price of Selected Diamond is $ {round(price[0],2)}")

        else:
            st.error('Please enter all values')

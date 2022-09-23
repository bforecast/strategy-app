from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import pandas as pd
import streamlit as st


data = {'cpu': ['Intel Core i7-12700K', 'Intel Core i9-12900K',
                'Intel Core i9-10850K', 'Intel Core i5-11400F'],
        'price': [350, 560, 300, 160]}


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


df = pd.DataFrame(data)

gb = GridOptionsBuilder.from_dataframe(df)
# gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_selection(selection_mode="single")
# gb.configure_side_bar()
gridoptions = gb.build()

response = AgGrid(
    df,
    height=200,
    gridOptions=gridoptions,
    enable_enterprise_modules=True,
    update_mode=GridUpdateMode.MODEL_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=True,
    header_checkbox_selection_filtered_only=True,
)

# st.write(type(response))
# st.write(response.keys())

v = response['selected_rows']
if v:
    st.write('Selected rows')
    st.dataframe(v)
    dfs = pd.DataFrame(v)
    csv = convert_df(dfs)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='selected.csv',
        mime='text/csv',
    )

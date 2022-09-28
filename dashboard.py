import pandas as pd
import numpy as np
import datetime, pytz

import streamlit as st
st.set_page_config(initial_sidebar_state='collapsed')

import psycopg2, psycopg2.extras
import vectorbt as vbt
from vectorbt.utils.colors import adjust_opacity

from bokeh.models import ColumnDataSource, CustomJS, DateFormatter, NumberFormatter
from bokeh.models import DataTable, TableColumn
from bokeh.plotting import figure
from streamlit_bokeh_events import streamlit_bokeh_events

from utils.db import init_connection, load_symbols
from utils.vbt_nb import show_pf, plot_pf
from utils.component import check_password
from utils.portfolio import Portfolio
import config

def show_PortfolioTable(portfolio_df):
        portfolio_df = portfolio_df[['name', 'annual_return', 'sharpe_ratio', 'total_return', 'maxdrawdown', 'start_date', 'end_date']]
        cds = ColumnDataSource(portfolio_df)
        columns = [
                    TableColumn(field='name', title='Name', width=200),
                    TableColumn(field='annual_return', title='Annualized', width=120, formatter=NumberFormatter(format='0:.0%')),
                    TableColumn(field='sharpe_ratio', title='Sharpe Ratio', width=120, formatter=NumberFormatter(format='0.00')),
                    TableColumn(field='total_return', title='Total Return%', width=120, formatter=NumberFormatter(format='0.00')),
                    TableColumn(field='maxdrawdown', title='Max DD%', width=120, formatter=NumberFormatter(format='0.00')),
                    TableColumn(field='start_date', title='Start Date', width=120, formatter=DateFormatter(format='%Y-%m-%d')),
                    TableColumn(field='end_date', title='End Date', width=120, formatter=DateFormatter(format='%Y-%m-%d')),
                    # TableColumn(field='description', title='Parameters')
                    ]
        # define events
        cds.selected.js_on_change(
                "indices",
                CustomJS(
                    args=dict(source=cds),
                    code="""
                    document.dispatchEvent(
                        new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}})
                    )
                    """
                )
            )

        table = DataTable(source=cds, columns=columns, row_height=30, selectable="checkbox",
                                index_position = None, aspect_ratio='auto', scroll_to_selection=True, height=300)

        result = streamlit_bokeh_events(
                    bokeh_plot=table,
                    events="INDEX_SELECT",
                    key="foo",
                    refresh_on_update=True,
                    # debounce_time=10,
                    override_height=300
                )

        if result and result.get("INDEX_SELECT"):
            return result.get("INDEX_SELECT")["data"]
        else:
            return []

def main():
    st.header("Portfolio Board")
    selected_pfs = []
    portfolio = Portfolio()
    selected_pfs = show_PortfolioTable(portfolio.df)
    if len(selected_pfs) > 1:
        pf_df = pd.DataFrame()
        for pfid in selected_pfs:
            filename = portfolio.df.iloc[pfid].at['filename']
            pf = vbt.Portfolio.load(config.PORTFOLIO_PATH + '/' + filename)
            pf_df[portfolio.df.iloc[pfid].at['name']] = pf.returns()
        pf_df = pf_df.cumsum()
        pf_df.fillna(method='ffill', inplace=True)
        pf_df['mean'] = pf_df.mean(axis=1)
        st.line_chart(pf_df)

    elif len(selected_pfs) == 1 :
        selected_portfolio = selected_pfs[0]      
        if selected_portfolio > -1 and selected_portfolio in portfolio.df.index:
            st.info('Selected portfolio:    ' + portfolio.df.at[selected_portfolio, 'name'])
            param_dict = portfolio.df.at[selected_portfolio, 'param_dict']

            cols = st.columns(3 + len(param_dict))
            with cols[0]:
                st.metric('Annualized', "{0:.0%}".format(portfolio.df.at[selected_portfolio, 'annual_return']))
            with cols[1]:
                st.metric('Sharpe Ratio', '%.2f'% portfolio.df.at[selected_portfolio, 'sharpe_ratio'])
            with cols[2]:
                st.metric('Max DD %', '%.0f'% portfolio.df.at[selected_portfolio, 'maxdrawdown'])
            i = 3
            for k, v in param_dict.items():
                with cols[i]:
                    st.metric(k, v)
                i = i + 1


            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                showpf_bool = st.checkbox("Show the Portfolio")
            with col2:
                updatepf_bool = st.button('Update')
            with col3:
                deletepf_bool = st.button('Delete')

            if showpf_bool:
                show_pf(portfolio.df.loc[selected_portfolio, 'filename'])
            if updatepf_bool:
                if portfolio.update(portfolio.df.loc[selected_portfolio, 'id']):
                    st.success('Update portfolio Sucessfully.')
                else:
                    st.error('Fail to update portfolio.')
                st.experimental_rerun()
            if deletepf_bool:
                if portfolio.delete(portfolio.df.loc[selected_portfolio, 'id']):
                    st.success('Delete portfolio Sucessfully.')
                else:
                    st.error('Fail to delete portfolio.')
                selected_pfs = []
                st.experimental_rerun()
    else:
        if st.button("Update All"):
            update_bar = st.progress(0)
            num_portfolio = len(portfolio.df)
            for i in range(num_portfolio):
                st.write(f"update portfolio('{portfolio.df.loc[i, 'name']}')")
                if not portfolio.update(portfolio.df.loc[i, 'id']):
                    st.error(f"Fail to update portfolio('{portfolio.df.loc[i, 'name']}')")
                update_bar.progress(i / (num_portfolio-1))

            st.experimental_rerun()

if __name__ == "__main__":
    if check_password():
        main()

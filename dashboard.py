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

        table = DataTable(source=cds, columns=columns, row_height=30, 
                                index_position = None, aspect_ratio='auto', scroll_to_selection=True, height=300)

        result = streamlit_bokeh_events(
                    bokeh_plot=table,
                    events="INDEX_SELECT",
                    key="foo",
                    refresh_on_update=True,
                    debounce_time=0,
                    override_height=300
                )

        if result and result.get("INDEX_SELECT"):
            return result.get("INDEX_SELECT")["data"][0]
        else:
            return -1

def show_StockTable():
    symbols_df = load_symbols()
    st.table(symbols_df)

def main():
        st.header("Portfolio Board")
        portfolio = Portfolio()
        # portfolio.df = portfolio.df
        selected_portfolio = show_PortfolioTable(portfolio.df)
        if selected_portfolio > -1 and selected_portfolio in portfolio.df.index:
            st.info('Selected portfolio:    ' + portfolio.df.at[selected_portfolio, 'name'])
            col1, col2, col3, col4 = st.columns([1, 1, 1, 4])
            with col1:
                st.metric('Annualized', "{0:.0%}".format(portfolio.df.at[selected_portfolio, 'annual_return']))
            with col2:
                st.metric('Sharpe Ratio', '%.2f'% portfolio.df.at[selected_portfolio, 'sharpe_ratio'])
            with col3:
                st.metric('Max DD %', '%.0f'% portfolio.df.at[selected_portfolio, 'maxdrawdown'])
            with col4:
                st.text('Parameters')
                if portfolio.df.at[selected_portfolio, 'param_dict']:
                    st.text([(k,v) for k,v in (portfolio.df.at[selected_portfolio, 'param_dict']).items()])

            col1, col2, col3 = st.columns(3)
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
                    st.experimental_rerun()
                else:
                    st.error('Fail to Update portfolio.')
            if deletepf_bool:
                if portfolio.delete(portfolio.df.loc[selected_portfolio, 'id']):
                    st.success('Delete portfolio Sucessfully.')
                    st.experimental_rerun()
                else:
                    st.error('Fail to delete portfolio.')


if __name__ == "__main__":
    if check_password():
        main()

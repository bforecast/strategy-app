from tracemalloc import start
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, date


import streamlit as st

st.set_page_config(initial_sidebar_state='collapsed', layout='wide')

# import psycopg2, psycopg2.extras
import vectorbt as vbt

from bokeh.models import ColumnDataSource, CustomJS, DateFormatter, NumberFormatter
from bokeh.models import DataTable, TableColumn
from streamlit_bokeh_events import streamlit_bokeh_events

from utils.vbt import plot_pf, show_pffromfile
from utils.component import check_password, params_selector
from utils.portfolio import Portfolio
from vbt_strategy.PairTrade import pairtrade_pfs
from pages.Strategy import check_params
from utils.db import get_symbolname

import config

def bokehTable(portfolio_df):
        portfolio_df = portfolio_df[['name', 'annual_return','lastday_return', 'sharpe_ratio', 'total_return', 'maxdrawdown', 'symbols', 'end_date']]
        cds = ColumnDataSource(portfolio_df)
        columns = [
                    TableColumn(field='name', title='Name', width=200),
                    TableColumn(field='annual_return', title='Annualized', width=120, formatter=NumberFormatter(format='0:.0%', text_align='right')),
                    TableColumn(field='lastday_return', title='Lastday Return', width=120, formatter=NumberFormatter(format='0:.0%', text_align='right')),
                    TableColumn(field='sharpe_ratio', title='Sharpe Ratio', width=120, formatter=NumberFormatter(format='0.00', text_align='right')),
                    TableColumn(field='total_return', title='Total Return', width=120, formatter=NumberFormatter(format='0:.0%', text_align='right')),
                    TableColumn(field='maxdrawdown', title='Max DD', width=120, formatter=NumberFormatter(format='0:.0%', text_align='right')),
                    TableColumn(field='symbols', title='Symbols', width=120),
                    TableColumn(field='end_date', title='End Date', width=120,),
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

        table = DataTable(source=cds, columns=columns, row_height=33, selectable="checkbox",
                        index_position = None, aspect_ratio='auto', scroll_to_selection=True, height=500, width=900)

        result = streamlit_bokeh_events(
                    bokeh_plot=table,
                    events="INDEX_SELECT",
                    key="foo",
                    refresh_on_update=True,
                    # debounce_time=10,
                    override_height=500
                )
        if result and result.get("INDEX_SELECT"):
            idata = result.get("INDEX_SELECT")["data"]
            cds_df = cds.to_df()
            for i in idata:
                if i not in cds_df.index.values:
                    return []
            return cds_df.loc[idata, 'index'].to_list()
        else:
            return []

def show_PortfolioTable(portfolio_df):
    def stringlist_to_set(strlist: list):
        slist = []
        for sstr in strlist:
            for s in sstr.split(','):
                slist.append(s)
        slist = list(dict.fromkeys(slist))
        slist.sort()
        return(slist)
    
    def selectpf_bySymbols(df, symbols:list):
        ids = set()
        for i, row in df.iterrows():
            for s in row['symbols'].split(','):
                if s in symbols:
                    ids.add(i)
        return df.loc[ids,:]

    symbols = stringlist_to_set(portfolio_df['symbols'].values)
    if 'symbolsSel' not in st.session_state:
        st.session_state['symbolsSel'] = symbols

    with st.form(key='selectSymbols_form'):
        col1, col2 = st.columns([7, 1])
        with col1:
            sSel = st.multiselect("Please select symbols:", symbols, 
                                format_func=lambda x: get_symbolname(x)+x,
                                help='empty means all')
        with col2:
            st.text('')
            st.text('')
            if st.form_submit_button("Select"):
                if len(sSel) > 0:
                    st.session_state['symbolsSel'] = sSel
                else:
                    st.session_state['symbolsSel'] = symbols

    df = selectpf_bySymbols(portfolio_df, st.session_state['symbolsSel'])
    selectpf = bokehTable(df)
    return(selectpf)

def show_PortforlioDetail(portfolio_df, index):
    if index > -1 and (index in portfolio_df.index):
        st.info('Selected portfolio:    ' + portfolio_df.at[index, 'name'])
        param_dict = portfolio_df.at[index, 'param_dict']

        cols = st.columns([1, 1, 1, 1, 3])
        with cols[0]:
            st.metric('Annualized', "{0:.0%}".format(portfolio_df.at[index, 'annual_return']))
        with cols[1]:
            st.metric('Lastday Return', "{0:.1%}".format(portfolio_df.at[index, 'lastday_return']))
        with cols[2]:
            st.metric('Sharpe Ratio', '%.2f'% portfolio_df.at[index, 'sharpe_ratio'])
        with cols[3]:
            st.metric('Max DD', '{0:.0%}'.format(portfolio_df.at[index, 'maxdrawdown']))
        with cols[4]:
            st.markdown("**Parameters**")
            st.text(param_dict)
        st.markdown("**Description**")
        st.markdown(portfolio_df.at[index, 'description'], unsafe_allow_html=True)
        return True
    else:
        return False

def show_PortforlioYearly(pf_row):
    end_date = date.today()
    end_date = datetime(year=end_date.year, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
    symbolsDate_dict = {
            "market":   pf_row['market'],
            "symbols":  [],
            "start_date": end_date,
            "end_date": end_date,
        }
    
    strategyname = pf_row['strategy']
    
    # get the strategy class according to strategy name
    strategy_cli = getattr(__import__(f"vbt_strategy"), f"{strategyname}Strategy")
    strategy = strategy_cli(symbolsDate_dict)
    params = params_selector(strategy.param_def)

    pfYearly_df = pd.DataFrame()                                 
    for y in range(1, 10, 2):
        start_date = datetime(year=end_date.year-y, month=end_date.month, day=end_date.day, tzinfo=pytz.utc)
        st.write(start_date)
        symbolsDate_dict = {
            "market":   pf_row['market'],
            "symbols":  [pf_row['symbols']],
            "start_date": start_date,
            "end_date": end_date,
        }
        strategy = strategy_cli(symbolsDate_dict)

        if check_params(params):
            if strategy.maxSR(params, output_bool=False):
                st.text("Max Sharpe_Ratio's parameters:    " + str(strategy.param_dict))
                pfYearly_df = pfYearly_df.append({
                            "year": y,
                            'total_return': round(strategy.pf.stats('total_return')[0]/100.0, 2),
                            'lastday_return': round(strategy.pf.returns()[-1], 2),
                            'sharpe_ratio':  round(strategy.pf.stats('sharpe_ratio')[0], 2),
                            'maxdrawdown':   round(strategy.pf.stats('max_dd')[0]/100.0, 2),
                            'annual_return': round(strategy.pf.annualized_return(), 2)
                        }, ignore_index=True, )
        
    st.table(pfYearly_df)    


def main():
    st.header("Portfolio Board")
    selected_pfs = []
    portfolio = Portfolio()
    selected_pfs = show_PortfolioTable(portfolio.df)
    if len(selected_pfs) > 1:
        ##多portfolio比较
        value_df = pd.DataFrame()
        position_df = pd.DataFrame()
        for index in selected_pfs:
            pf = vbt.Portfolio.loads(portfolio.df.iloc[index].at['vbtpf'])
            value_df[portfolio.df.iloc[index].at['name']] = pf.value()
            position_df[portfolio.df.iloc[index].at['name']] = pf.position_mask()
            show_PortforlioDetail(portfolio.df, index)
        # value_df = value_df.cumsum()
        value_df.fillna(method='ffill', inplace=True)
        # value_df['mean'] = value_df.mean(axis=1)
        st.line_chart(value_df)
        st.plotly_chart(position_df.vbt.plot(), user_container_width = True)

        if len(selected_pfs) == 2:
            if st.button("PairTrade"):
                pf = pairtrade_pfs(value_df.columns[0], value_df.columns[1],
                                value_df.iloc[:,0], value_df.iloc[:,1], True)
                if pf:
                    plot_pf(pf)
                else:
                    st.error(f"Fail to PairTrade '{value_df.columns[0]}' and '{value_df.columns[1]}'.")
            
            if st.button("Master/Backup"):
                    symbol1 = value_df.columns[0]
                    symbol2 = value_df.columns[1]

                    symbol_cols = pd.Index([symbol1, symbol2], name='symbol')
                    # Build percentage order size
                    vbt_order_size = pd.DataFrame(index=value_df.index, columns=symbol_cols)
                    vbt_order_size[symbol1] = 0
                    vbt_order_size[symbol2] = 1
                    vbt_order_size.loc[position_df.iloc[:,0], symbol1] = 1
                    vbt_order_size.loc[position_df.iloc[:,0], symbol2] = 0
                    # Execute at the next bar
                    vbt_order_size = vbt_order_size.vbt.fshift(1)

                    vbt_close_price = pd.concat((value_df.iloc[:,0], value_df.iloc[:,1]), axis=1, keys= symbol_cols)

                    pf_kwargs = dict(fees=0.001, freq='1D')
                    ms_pf = vbt.Portfolio.from_orders(
                                                    vbt_close_price,  # current close as reference price
                                                    size=vbt_order_size,  
                                                    price=vbt_close_price,  # current open as execution price
                                                    size_type='targetpercent', 
                                                    init_cash=100,
                                                    cash_sharing=True,  # share capital between assets in the same group
                                                    group_by=True,  # all columns belong to the same group
                                                    call_seq='auto',  # sell before buying
                                                    **pf_kwargs)
                    plot_pf(ms_pf)
                    value_df["Master/Backup"] = ms_pf.value()
                    st.line_chart(value_df)



    elif len(selected_pfs) == 1 :
        ##单portforlio详情
        index = selected_pfs[0]      
        if show_PortforlioDetail(portfolio.df, index):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                showpf_bool = st.checkbox("Show the Portfolio")
            with col2:
                morepf_bool = st.button('More')      
            with col3:
                updatepf_bool = st.button('Update')
            with col4:
                deletepf_bool = st.button('Delete')

            if showpf_bool:
                show_pffromfile(portfolio.df.loc[index, 'vbtpf'])
            if morepf_bool:
                show_PortforlioYearly(portfolio.df.iloc[index, :])
            if updatepf_bool:
                if portfolio.update(portfolio.df.loc[index, 'id']):
                    st.success('Update portfolio Sucessfully.')
                else:
                    st.error('Fail to update portfolio.')
                st.experimental_rerun()
            if deletepf_bool:
                if portfolio.delete(portfolio.df.loc[index, 'id']):
                    st.success('Delete portfolio Sucessfully.')
                else:
                    st.error('Fail to delete portfolio.')
                st.experimental_rerun()
        else:
            selected_pfs = []

    else:
        ##无选择portforlio
        if st.button("Update All"):
            update_bar = st.progress(0)
            num_portfolio = len(portfolio.df)
            info_holder = st.empty()
            for i in range(num_portfolio):
                info_holder.write(f"updating portfolio('{portfolio.df.loc[i, 'name']}')")
                if not portfolio.update(portfolio.df.loc[i, 'id']):
                    st.error(f"Fail to update portfolio('{portfolio.df.loc[i, 'name']}')")
                update_bar.progress(i / (num_portfolio-1))
                info_holder.empty()

            st.experimental_rerun()

if __name__ == "__main__":
    if check_password():
        main()

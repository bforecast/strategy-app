CREATE TABLE stock (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    is_etf BOOLEAN NOT NULL
);

CREATE TABLE mention (
    stock_id INTEGER,
    dt TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    message TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    username TEXT NOT NULL,
    PRIMARY KEY (stock_id, dt),
    CONSTRAINT fk_mention_stock FOREIGN KEY (stock_id) REFERENCES stock (id)
);

CREATE INDEX ON mention (stock_id, dt DESC);
SELECT create_hypertable('mention', 'dt');

CREATE TABLE etf_holding (
    etf_id INTEGER NOT NULL, 
    holding_id INTEGER NOT NULL,
    dt DATE NOT NULL, 
    shares NUMERIC,
    weight NUMERIC, 
    PRIMARY KEY (etf_id, holding_id, dt),
    CONSTRAINT fk_etf FOREIGN KEY (etf_id) REFERENCES stock (id),
    CONSTRAINT fk_holding FOREIGN KEY (holding_id) REFERENCES stock (id)
);

CREATE TABLE stock_price (
    stock_id INTEGER NOT NULL,
    dt TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    open NUMERIC NOT NULL, 
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL, 
    volume NUMERIC NOT NULL,
    PRIMARY KEY (stock_id, dt),
    CONSTRAINT fk_stock FOREIGN KEY (stock_id) REFERENCES stock (id)
);

CREATE INDEX ON stock_price (stock_id, dt DESC);

SELECT create_hypertable('stock_price', 'dt');


CREATE MATERIALIZED VIEW daily_bars
WITH (timescaledb.continuous) AS
SELECT stock_id,
       time_bucket(INTERVAL '1 day', dt) AS day,
       first(open, dt) as open,
       max(high) as high,
       min(low) as low,
       last(close, dt) as close,
       sum(volume) as volume
FROM stock_price
GROUP BY stock_id, day;


CREATE MATERIALIZED VIEW hourly_bars
WITH (timescaledb.continuous) AS
SELECT stock_id,
       time_bucket(INTERVAL '1 hour', dt) AS hour,
       first(open, dt) as open,
       max(high) as high,
       min(low) as low,
       last(close, dt) as close,
       sum(volume) as volume
FROM stock_price
GROUP BY stock_id, hour;

CREATE TABLE portfolio (
    id SERIAL PRIMARY KEY,
    create_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    start_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    end_date TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    total_return NUMERIC NOT NULL, 
    today_return NUMERIC,
    annual_return NUMERIC NOT NULL, 
    sharpe_ratio NUMERIC NOT NULL, 
    maxdrawdown  NUMERIC NOT NULL,
    filename TEXT NOT NULL,
    param_dict JSONB,
    strategy TEXT NOT NULL,
    symbols TEXT NOT NULL,
    market TEXT NOT NULL
);

CREATE TABLE portfolio_stock (
    portfolio_id INTEGER NOT NULL, 
    stock_id INTEGER NOT NULL,
    PRIMARY KEY (portfolio_id, stock_id),
    CONSTRAINT fk_portfolio FOREIGN KEY (portfolio_id) REFERENCES portfolio (id),
    CONSTRAINT fk_stock FOREIGN KEY (stock_id) REFERENCES stock (id)
);
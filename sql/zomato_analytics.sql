-- =============================================================================
-- zomato_analytics.sql
-- Analytics queries against the SQLite in-memory DB loaded by sql_runner.py
-- =============================================================================

-- -------------------------------------------------------------------------
-- 1. Cost Tier Classification (CASE WHEN)
-- -------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_cost_tiers AS
SELECT
    restaurant_id,
    name,
    location,
    rate,
    votes,
    "approx_cost(for two)" AS cost,
    online_order,
    book_table,
    cuisines,
    "listed_in(type)"                       AS listing_type,
    CASE
        WHEN "approx_cost(for two)" < 300   THEN 'Budget'
        WHEN "approx_cost(for two)" < 700   THEN 'Mid-Range'
        WHEN "approx_cost(for two)" < 1200  THEN 'Premium'
        ELSE                                     'Luxury'
    END                                     AS cost_tier
FROM restaurants;


-- -------------------------------------------------------------------------
-- 2. Window Function – DENSE_RANK by votes within each location
-- -------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_ranked_by_location AS
SELECT
    restaurant_id,
    name,
    location,
    rate,
    votes,
    cost_tier,
    DENSE_RANK() OVER (
        PARTITION BY location
        ORDER BY votes DESC
    )                                       AS rank_in_location,
    NTILE(4) OVER (
        ORDER BY votes DESC
    )                                       AS votes_quartile,
    ROUND(AVG(rate) OVER (
        PARTITION BY location
    ), 2)                                   AS avg_rating_in_location,
    ROUND(AVG("approx_cost(for two)") OVER (
        PARTITION BY location
    ), 0)                                   AS avg_cost_in_location
FROM v_cost_tiers;


-- -------------------------------------------------------------------------
-- 3. CTE – Top restaurants per location (rank = 1)
-- -------------------------------------------------------------------------
WITH top_per_location AS (
    SELECT *
    FROM   v_ranked_by_location
    WHERE  rank_in_location = 1
)
SELECT
    t.location,
    t.name                      AS top_restaurant,
    t.votes,
    t.rate,
    t.cost_tier,
    t.avg_rating_in_location,
    t.avg_cost_in_location
FROM   top_per_location t
ORDER  BY t.votes DESC
LIMIT  30;


-- -------------------------------------------------------------------------
-- 4. CTE – Online-order adoption rate by location
-- -------------------------------------------------------------------------
WITH online_stats AS (
    SELECT
        location,
        COUNT(*)                                                AS total,
        SUM(CASE WHEN online_order = 'Yes' THEN 1 ELSE 0 END)  AS online_count
    FROM   restaurants
    GROUP  BY location
)
SELECT
    location,
    total,
    online_count,
    ROUND(100.0 * online_count / total, 1)  AS online_pct
FROM   online_stats
ORDER  BY online_pct DESC;


-- -------------------------------------------------------------------------
-- 5. CTE – Rating band distribution with running totals
-- -------------------------------------------------------------------------
WITH bands AS (
    SELECT
        CASE
            WHEN rate < 3.0 THEN '1-Poor (<3)'
            WHEN rate < 3.5 THEN '2-Fair (3-3.5)'
            WHEN rate < 4.0 THEN '3-Good (3.5-4)'
            WHEN rate < 4.5 THEN '4-Very Good (4-4.5)'
            ELSE                 '5-Excellent (>=4.5)'
        END                 AS rating_band,
        COUNT(*)            AS restaurant_count
    FROM   restaurants
    GROUP  BY rating_band
),
running AS (
    SELECT
        rating_band,
        restaurant_count,
        SUM(restaurant_count) OVER (ORDER BY rating_band)  AS running_total
    FROM   bands
)
SELECT * FROM running ORDER BY rating_band;

Market Order TCA Framework
Core question: where do we see higher or lower arrival-to-execution slippage for market orders for 2026H1?
Slippage definition:
Cost bps = Side Sign × (Execution Average Price - Arrival Price) / Arrival Price × 10,000
    where Buy  = +1, Sell = -1
Data fields
Order data: Trade date, Arrival time, Arrival price, Direction, Broker
Market data: Arrival price, Average price, Market Cap, Region, Industry
Cost calculations:  Mean Cost (bps), Median Cost (bps), Std Dev Cost (bps), Cost t-stat
Other data for further analysis: AVAT, Historical Volatility, Interval VWAP
Source of Trading Cost and Grouping Methodology
Note: Grouping methods below are initial proposals, and we can try different definitions for market cap group, ADV% group, spread group; we could even explore different sets of features combinations to group by. We could even run machine learning algorithm to decide feature significance after proper data cleaning when we have sufficient sample sizes.
Broker Quality: trading quality of broker algorithms and execution by region/industry, etc.
- Group by: Region -> Industry -> Broker -> Direction
Size effect: cost from small cap trading.
- Group by: Market Cap Group -> Direction

Market impact: cost from high market participation.
-  Group by: ADV% Group -> Direction

Liquidity cost: cost from less liquidity providers and large spread crossing.
- Group by: bid/ask spread (bps) -> Direction

Spread Bucket
	
Definition


Tight
	
0 – 10 bps


Medium
	
10 – 20 bps


Wide
	
20 – 50 bps


Very Wide
	
> 50 bps

ADV%
	
Definition


Small
	
< 1% ADV


Medium
	
1% – 5% ADV


Large
	
5% – 10% ADV


Very Large
	
> 10% ADV

 
Market Cap Group
	
Definition


Large Cap
	
> USD 10bn


Mid Cap
	
USD 2bn – 10bn


Small Cap
	
< USD 2bn
 
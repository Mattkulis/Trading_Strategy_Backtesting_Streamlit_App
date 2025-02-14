# Trading_Strategy_Backtesting_Streamlit_App
# Link:  https://mattkulisbacktestapp.streamlit.app/

Ensure dependancies and python env downloaded & ensure that you are running the most up to date version. As I write this it is currently v6.8
#The images are ytd, trailing 12mo, & 2year lookback periods for reference of strategy performance.

![image](https://github.com/user-attachments/assets/9c6d2b91-eace-4b32-89ff-8d09345b6cd8)
![image](https://github.com/user-attachments/assets/6b11c843-9903-4f9d-9fb7-b0f0f0ef534b)



Entry Criteria:

Current candle must be green (close > open),
Close must be greater than 9 EMA,
Current candle's close must be higher than the opens of the previous 6 candles,
Current volume must be greater than the maximum volume of any red candles in the previous 6 candles.

Exit Criteria:

Either 3 consecutive red candles OR 2 consecutive red candles AND close below 9 EMA
Execution price calculation: using the average (open + high + low + close) / 4

Position sizing:

Initial position is 100 shares. Program can add 100 shares up to 300 total if momentum continues.

Program exits entire position when exit conditions are met.

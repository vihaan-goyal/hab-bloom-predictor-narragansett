import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/hab_labels_lis.csv")
df['time'] = pd.to_datetime(df['time'], utc=True)
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month

# Bloom rate by year
yearly = df.groupby('year').agg(
    total=('bloom', 'count'),
    blooms=('bloom', 'sum')
)
yearly['bloom_rate'] = yearly['blooms'] / yearly['total'] * 100

# Bloom rate by month
monthly = df.groupby('month').agg(
    total=('bloom', 'count'),
    blooms=('bloom', 'sum')
)
monthly['bloom_rate'] = monthly['blooms'] / monthly['total'] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.bar(yearly.index, yearly['bloom_rate'], color='green', alpha=0.7)
ax1.set_title('Bloom frequency by year (% readings > 10 ug/L)')
ax1.set_ylabel('% bloom readings')
ax1.set_xlabel('Year')

ax2.bar(monthly.index, monthly['bloom_rate'], color='teal', alpha=0.7)
ax2.set_title('Bloom frequency by month')
ax2.set_ylabel('% bloom readings')
ax2.set_xlabel('Month')
ax2.set_xticks(range(1,13))
ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.tight_layout()
plt.savefig('figures/eda/bloom_frequency.png', dpi=150)
plt.show()
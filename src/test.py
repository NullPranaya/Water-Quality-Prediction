

# Lowercase, strip spaces, replace gaps with underscores
df_wq.columns = df_wq.columns.str.strip().str.replace(" ", "_")

# Add _ before any capital letters, excluding the first letter


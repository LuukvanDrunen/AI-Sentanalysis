library(ggplot2)
library(dplyr)
library(scales)

# Load data
df <- read.csv("Bontenbal_labeled_with_dates.csv",
               stringsAsFactors = FALSE, sep=';')

# Parse date and extract year-month
df$date <- as.POSIXct(df$date_created, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")
df$year_month <- as.Date(format(df$date, "%Y-%m-01"))

# Aggregate: monthly proportion and counts
monthly <- df %>%
  group_by(year_month) %>%
  summarise(
    n_total   = n(),
    n_label1  = sum(Labels == 0),
    prop_1    = mean(Labels == 0),
    .groups   = "drop"
  ) %>%
  filter(n_total >= 5)  # drop months with very few posts

# Smooth trend via loess
fit <- loess(prop_1 ~ as.numeric(year_month), data = monthly, span = 0.35)
monthly$trend <- predict(fit)

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggplot(monthly, aes(x = year_month)) +

  # Stacked bar: proportion 0 (bottom) and 1 (top)
  geom_col(aes(y = 1), fill = "#c8d8e8", width = 25, alpha = 0.6) +
  geom_col(aes(y = prop_1), fill = "#2166ac", width = 25, alpha = 0.75) +

  # Raw proportion dots
  geom_point(aes(y = prop_1, size = n_total),
             colour = "#1a4f7a", alpha = 0.7, shape = 21, fill = "#4393c3") +

  # LOESS trend line
  geom_line(aes(y = trend), colour = "#d6604d", linewidth = 1.1, linetype = "solid") +

  # Reference line at 50 %
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40", linewidth = 0.5) +

  scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y",
    expand = c(0.02, 0)
  ) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  scale_size_continuous(
    name   = "Posts\nper month",
    range  = c(1.5, 7),
    breaks = c(10, 50, 100, 200)
  ) +

  labs(
    title    = "Share of Label-0 Posts Over Time (Henri Bontenbal)",
    subtitle = "Bars = monthly proportion  |  Red line = LOESS trend  |  Dot size = post volume",
    x        = NULL,
    y        = "Proportion labelled 0",
    caption  = "Dashed line at 50 %"
  ) +

  theme_minimal(base_size = 13) +
  theme(
    plot.title       = element_text(face = "bold", size = 15),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x      = element_text(angle = 30, hjust = 1),
    legend.position  = "right"
  )


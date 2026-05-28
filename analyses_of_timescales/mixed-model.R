# @author: Ana Antonia Dias Maile

library(dplyr)
library("lme4")
library("lmerTest")
library("car")
library("sjPlot")
library(ggplot2)

#clear workspace
rm(list = ls())

###################
## PREPARE PATHS ##
###################
setwd('../../../')
path <- getwd()

###############
## LOAD DATA ##
###############
# load fits
fits = read.csv("spectral_fits_per_vertex.csv")
fits = fits[,3:ncol(fits)]
# load t1t2
t1t2 = read.csv("t1t2values.csv")
t1t2 = t1t2[,2]
# load label order
label_order_fsLR = read.csv("label_order_fsLR.csv")
label_order_fsLR = label_order_fsLR[,2]
# load control parameters
setwd("../../")
sessiontsv <- read.csv("session_tsv_v2.csv")

##################
## PREPARE DATA ##
##################
# timescales
# median over vertex
time_constant <- fits %>%
  filter(!is.na(label), label != "") %>% # take out vertex that were not assigned a label
  group_by(sub, drug, ses, label) %>%
  summarise(time_constant = median(time_constant, na.rm = TRUE), .groups = "drop")
# order according to label_order_fsLR
time_constant <- time_constant %>%
  mutate(label = factor(label, levels = label_order_fsLR)) %>%
  arrange(sub, drug, label)
# add t1t2 values
time_constant<- time_constant %>%
  mutate(t1t2 = t1t2[match(label, label_order_fsLR)])
# make contrasts
time_constant$drug_f <- factor(time_constant$drug)
contrasts(time_constant$drug_f) <- contr.treatment(3,base = 3)
# logarithmic transformation
time_constant$time_constant_log <- log(time_constant$time_constant)

#################
## MIXED MODEL ##
#################
# how to compute pvalue
options(lmerTest.default.method = "Kenward-Roger")
# mixed model
model <- lmer(time_constant_log ~ t1t2*drug_f + (1 + t1t2+drug_f| sub), data = time_constant)
summary(model)
# plot
cols <- c("#66C2A5", "#B3B3B3", "#FC8D62")
interact_plot(model_time,
              pred = t1t2,
              modx = drug_f,
              interval = TRUE,
              plot.points = FALSE,
              vary.lty = FALSE,
              dodge = 0.15,
              errorbar.width = 0.01,
              pred.point.size = 1.9,
              line.thickness = 0.5,
              geom.alpha = 0.1,
              y.transform = exp) +
  geom_vline(xintercept = 0, color = "grey", linetype = "dotted", size = 0.5) +
  labs(x = "T1–T2 Condition",
       y = "Time Constant (real scale)",
       color = "Drug",
       fill = "Drug") +
  scale_colour_manual(labels = c("Lorazepam", "Placebo", "D-cycloserine"),
                      values = cols) +
  scale_fill_manual(labels = c("Lorazepam", "Placebo", "D-cycloserine"),
                    values = alpha(cols, 0.1)) +
  theme_minimal(base_size = 18) +
  theme(plot.title = element_blank(),
        strip.text.x = element_text(size = 22),
        strip.background = element_blank(),
        axis.text.x = element_text(color = "black", size = 22),
        axis.text.y = element_text(color = "black", size = 22),
        axis.title.x = element_text(size = 22, vjust = 0),
        axis.title.y = element_text(size = 22, vjust = +2),
        axis.line.x = element_line(color = "black", size = 0.5),
        axis.line.y = element_line(color = "black", size = 0.5),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

######################
## CONTROL ANALYSES ##
######################
# control for session effects
model_ses <- lmer(time_constant_log ~ scale(t1t2)*drug_f +
                    scale(ses)*scale(t1t2) + (1 + scale(t1t2)+drug_f| sub),
                  data = time_constant)
summary(model_ses)
# control for systolic blood pressure effects
sys <- sessiontsv[,c("participant_id", "session_id", "drug_label",
                     "BP_syst_time.1", "BP_syst_time.2")]
names(sys) <- c("sub", "ses", "drug", "sys_1", "sys_2")
sys$sys_diff <- sys$sys_1 - sys$sys_2
time_constant_sys <- merge(time_constant, sys, by = c('sub', 'ses', 'drug'))
model_sys <- lmer(time_constant_log ~ scale(t1t2)*drug_f + scale(sys_diff)*
                    drug_f + (1 + scale(t1t2)+drug_f| sub),
                  data = time_constant_sys)
summary(model_sys)
# control for heart rate effects
hr <- sessiontsv[,c("participant_id", "session_id", "drug_label",
                    "heart_rate_time.1", "heart_rate_time.2")]
names(hr) <- c("sub", "ses", "drug", "hr_1", "hr_2")
hr$hr_diff <- hr$hr_1 - hr$hr_2
time_constant_hr <- merge(time_constant, hr, by = c('sub', 'ses', 'drug'))
model_hr <- lmer(time_constant_log ~ scale(t1t2)*drug_f +
                        scale(hr_diff)*drug_f + (1 + scale(t1t2)+drug_f| sub),
                      data = time_constant_hr)
summary(model_hr)

# control for alertness effects
alert <- sessiontsv[,c("participant_id", "session_id", "drug_label",
                       "BL_VAS_alertness_time.1", "BL_VAS_alertness_time.2")]
names(alert) <- c("sub", "ses", "drug", "alert_1", "alert_2")
alert$alert_diff <- alert$alert_1 - alert$alert_2
time_constant_alert <- merge(time_constant, alert, by = c('sub', 'ses', 'drug'))
model_time_alert <- lmer(time_constant_log ~ scale(t1t2)*drug_f +
                           scale(alert_diff)*drug_f + (1 + scale(t1t2)+drug_f| sub),
                         data = time_constant_alert)
summary(model_time_alert)

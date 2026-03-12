###########LIBRARIES#####
#install.packages("ROCR")
library(ggplot2)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(ROCR)
library(randomForest)
library(pROC)


###########TAREA MODELO CLASIFICACIONES ###########
#1ra visualizacion de los datos#
df <- read.csv("churn-analysis.csv", sep = ";", header = TRUE)
df %>% str
df %>% head

#Nuestra data esta desbalanceada
#Si el modelo siempre dice falso acierta el 85.5% de las veces.
table(df$churn)
prop.table(table(df$churn))

table(df$area.code)
#Vemos que solo hay 3 códigos por lo que puede servirnos como forma de estudiar si el area afecta.

#sacamos los nuemero de telefono pues no nos sirven
#1.el numero de telefono no implica nada en la desición de permanecer en la compañía
#2.Son demasiados estados y en realidad no aportan ninguna información indispensable, y si fuera así tenemos las areas
#3.total.*.charge y total.*.minute estan fuertemente correlacionadas por lo que en realidad para predecir el comportamiento de los usuarios solo necesitamo uno (minutes)
df_c <- df %>%
  select(-phone.number,
         -state,
         -total.day.charge,
         -total.eve.charge,
         -total.night.charge,
         -total.intl.charge)

df_c %>% str
df_c %>% summary
#Qué podemos ver del summary:
#1.- 5 variables categóricas
#2.- puede que tengamos outliers en algunas variables

#Pasamos a factor las variables categoricas del df  
df_c <- df_c %>%
  mutate(
    international.plan = as.factor(international.plan),
    voice.mail.plan = as.factor(voice.mail.plan),
    churn = as.factor(churn),
    area.code = as.factor(area.code)
  )
#1 = no
#2 = yes

##########  ANALISIS EXPLORATORIO  ##########
df_c %>% summary
#vemos que:
#1.pocas personas tienen plan internacional
#2.pocas personas tienen correo de voz
#3.pocas personas se van de la compañía, por lo que se nos puede hacer dificil predecir cuando alguien se vaya.

#Servicio al cliente
ggplot(data = df_c, aes(x = customer.service.calls)) +
  geom_histogram() +
  labs(x ='Cantidad de llamadas a la mesa de ayuda', y='Frecuencia')
ggplot(df_c, aes(x = churn, y = customer.service.calls, fill = churn)) +
  geom_boxplot() +
  labs(x = "El cliente se va de la compañía",
       y = "N° de llamadas a la mesa de ayuda") +
  scale_fill_manual(values = c("False" = "green", "True" = "red"))
#vemos que quienes llaman más de una vez al servicio al cliente tienden a irse de la compañía.
#vemos outlierse en los que se quedan en la compañía!!



#Minutos por cliente
ggplot(data = df_c, aes(x = total.day.minutes)) +
  geom_histogram(binwidth = 20) +
  labs(x ='Total de minutos diarios', y='Frecuencia')
ggplot(data = df_c, aes(x = total.eve.minutes)) +
  geom_histogram(binwidth = 20) +
  labs(x ='Total de minutos por la tarde', y='Frecuencia')
ggplot(data = df_c, aes(x = total.night.minutes)) +
  geom_histogram(binwidth = 20) +
  labs(x ='Total de minutos por la noche', y='Frecuencia')
#Podemos ver valores muy similares en número de minutos usado por los usuarios en mañana, tarde y noche.
df_c %>%
  group_by(churn) %>%
  summarise(media_dia = mean(total.day.minutes),
            media_tarde = mean(total.eve.minutes),
            media_noche = mean(total.night.minutes))

#Podemos ver que lo normal es que los usuarios tengan un uso diario de 200 minutos
#Será que los que se fugan presentan un comportamiento diferente?

#Estudiamos boxplots
ggplot(df_c, aes(x = churn, y = total.day.minutes, fill = churn)) +
  geom_boxplot() +
  labs(x = "El cliente se va de la compañía",
       y = "Total day minutes") +
  scale_fill_manual(values = c("False" = "green", "True" = "red"))
#En rojo vemos quienes se fueron de la compañía y en verde los que se quedan en la compañía
#Varios outliers dentro de la categoría de los que se quedan en la compañía
#Tanto en el día, tarde y noche, las medianas son similares para los usuraios que se quedan y que se fugan
#Vemos cómo quienes se fugan presentan una media superior en total de minutos al día que quienes se quedan en la compañía
#Esto puede estar asociado a un gasto mayor en minutos == mayor cantidad de dinero gastada por el usario.
#Quienes consumen más diariamente tienden a irse
####
ggplot(df_c, aes(x = international.plan, fill = churn)) +
  geom_bar(position = "fill") +
  labs(y = "Proporción", x = "Plan internacional") +
  scale_fill_manual(values = c("False" = "green", "True" = "red"))

table(df$international.plan)
prop.table(table(df$international.plan))

#No hay valores NA en la data
colSums(is.na(df))
boxplot(df_c$total.day.minutes, main = "Total minutos en día")
boxplot(df_c$customer.service.calls , main = "Llamadas a mesa de ayuda")
#Se pueden visualizar valores atípicos 


########## MODELO PROPUESTO ########### 

###ARBOL DE DECIONES ###
####Primero separamos el train data y el test data.
set.seed(123)
train_idx <- sample(1:nrow(df_c),
              round(nrow(df_c)*0.7))
train <- df_c[train_idx,]
test <- df_c[-train_idx,]

#Verificamos la distribucion del conjunto de prueba y de entrenamiento
train %>% 
  group_by(churn) %>% 
  summarise(n = n(), porcentaje = (n / nrow(train)) * 100)
test %>% 
  group_by(churn) %>% 
  summarise(n = n(), porcentaje = (n / nrow(test)) * 100)

#Comenzamos a entrenar el modelo de arbol de decisiones
fit_class = rpart(churn ~ .,
                  data = train,
                  method="class",
                  control = list(cp=0.00001))
rpart.plot(fit_class,cex=0.5)

train %>% head
fit_class$variable.importance #Importancia de cada variable
#Probabilidad que el cliente se fuge
pred_prob <- predict(fit_class, newdata = test, type = "prob")[,2]

#Debido a lo desbalanceado que están los datos se considera pertinente un umbral de 0.3
pred_class <- ifelse(pred_prob > 0.3, "True", "False")
pred_class <- as.factor(pred_class)

#Confusion Matrix
conf_matrix <- confusionMatrix(pred_class, test$churn)
# Accuracy
conf_matrix$overall["Accuracy"]
# Precision 
conf_matrix$byClass["Precision"]
# Sensitivity (Recall)
conf_matrix$byClass["Sensitivity"]
# F1 Score
conf_matrix$byClass["F1"]
# Balanced Accuracy
conf_matrix$byClass["Balanced Accuracy"]
#Crear objeto de predicción
roc_pred <- prediction(pred_prob, test$churn)
# Métricas ROC
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
# Graficar ROC
plot(roc_perf,
     colorize = TRUE,
     text.adj = c(-0.2, 1.7),
     print.cutoffs.at = seq(0, 1, 0.1))
abline(a = 0, b = 1, col = "brown")
# Calcular AUC
auc <- performance(roc_pred, measure = "auc")@y.values[[1]]
auc

#En general el modelo no es malo pero puede ser mejor, por lo que podamos el arbol:
#PODAMOS EL ARBOL
#primero buscamos un cp óptimo
cp_opt <- fit_class$cptable[which.min(fit_class$cptable[,"xerror"]), "CP"]
fit_pruned <- prune(fit_class, cp = cp_opt)
rpart.plot(fit_pruned, cex = 0.5)

#Estudiamos ahora el árbol podado:
pred_prob2 <- predict(fit_pruned, newdata = test, type = "prob")[,2]

# Clase predicha usando umbral 0.5
pred_class2 <- ifelse(pred_prob2 > 0.5, "True", "False")
pred_class2 <- as.factor(pred_class2)


# Matriz de confusión
conf_matrix_pruned <- confusionMatrix(pred_class2, test$churn, positive = "True")
# Accuracy
conf_matrix_pruned$overall["Accuracy"]
# Precision (Positive Predictive Value)
conf_matrix_pruned$byClass["Precision"]
# Sensitivity (Recall)
conf_matrix_pruned$byClass["Sensitivity"]
# F1 Score
conf_matrix_pruned$byClass["F1"]
# Balanced Accuracy
conf_matrix_pruned$byClass["Balanced Accuracy"]

#Crear objeto de predicción
roc_pred_pruned <- prediction(pred_prob2, test$churn)
# Métricas ROC
roc_perf_pruned <- performance(roc_pred_pruned, measure = "tpr", x.measure = "fpr")
# Graficar ROC
plot(roc_perf_pruned,
     colorize = TRUE,
     text.adj = c(-0.2, 1.7),
     print.cutoffs.at = seq(0, 1, 0.1))
abline(a = 0, b = 1, col = "brown")
# Calcular AUC
auc <- performance(roc_pred_pruned, measure = "auc")@y.values[[1]]
auc



#####RANDOM FOREST. ####

# Entrenamos el modelo Random Forest
modelo_rf <- randomForest(churn ~ .,
                          data = train,
                          mtry = 7,        # cantidad de variables aleatorias por split
                          ntree = 200,      # cantidad de árboles
                          importance = TRUE)
# Gráfico del error vs cantidad de árboles
modelo_rf
plot(modelo_rf)

# IMPORTANCIA DE VARIABLES
importancia_pred <- as.data.frame(importance(modelo_rf, scale = TRUE))
importancia_pred <- rownames_to_column(importancia_pred, var = "variable")

p1 <- ggplot(data = importancia_pred, aes(x = reorder(variable, `MeanDecreaseGini`),
                                          y = `MeanDecreaseGini`,
                                          fill = `MeanDecreaseGini`)) +
  labs(x = "Variable", y = "Importancia", title = "Importancia de variables (Gini)") +
  geom_col() +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none")
p1

rf_pred_prob <- predict(modelo_rf, newdata = test, type = "prob")[, "True"]
rf_pred_class <- as.factor(ifelse(rf_pred_prob > 0.4, "True", "False"))
conf_matrix_rf <- confusionMatrix(rf_pred_class, test$churn, positive = "True")

#Curva ROC
roc_pred_rf <- prediction(rf_pred_prob, test$churn)
roc_perf_rf <- performance(roc_pred_rf, measure = "tpr", x.measure = "fpr")

# Graficar curva ROC
plot(roc_perf_rf,
     colorize = TRUE,
     text.adj = c(-0.2, 1.7),
     print.cutoffs.at = seq(0, 1, 0.1))
abline(a = 0, b = 1, col = "brown")

#AUC
auc <- performance(roc_pred_rf, measure = "auc")@y.values[[1]]
auc
  
conf_matrix_rf$overall["Accuracy"]
# Precision (Positive Predictive Value)
conf_matrix_rf$byClass["Precision"]
# Sensitivity (Recall)
conf_matrix_rf$byClass["Sensitivity"]
# F1 Score
conf_matrix_rf$byClass["F1"]
# Balanced Accuracy
conf_matrix_rf$byClass["Balanced Accuracy"]






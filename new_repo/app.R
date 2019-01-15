# password protect nginix server
# add ngram slide bar
# add top words slider bar
# add privacy policy, and terms of use

# add jpeg download
# run app from non root




######################  ######################  ######################   clean env
# rm(list = ls())

######################  ######################  ######################   temp var testing only
# n_topics <- 10


######################  ######################  ######################   load libs

library(tidytext) #text mining, unnesting # ms
library(dplyr) #awesome tools # ms
library(tm) #text mining # ms
library(stringr) # ms
library(text2vec) # ms
library(LDAvis) # ms
library(tokenizers) # ms
library(textstem) # ms
library(qdapTools) # ms
library(qdapRegex) # ms
library(shiny) # ms
library(shinydashboard) # ms
# library(shiny.semantic)
# library(semantic.dashboard)
# library(shinythemes)

library(qdap) # ms
library(plotly) # ms
# library(radarchart)
library(rebus) # ms
library(stringi)

######################  ###################### ###################### load app test data

ms_df_pos <- read.csv("data/ms_df_pos.csv", stringsAsFactors = FALSE)
ms_df_cons <- read.csv("data/ms_df_cons.csv", stringsAsFactors = FALSE)
gs_df_pos <- read.csv("data/gs_df_pos.csv", stringsAsFactors = FALSE)
gs_df_cons <- read.csv("data/gs_df_cons.csv", stringsAsFactors = FALSE)


######################  ###################### ###################### sentence tokenizer function
func_sent_tokenize <- function(df_text, st_func_activate=TRUE) {
  
  if (st_func_activate) {
    
    # # utf8 conversion
    # df_text$text_dm <- stri_enc_toutf8(df_text$text_dm)
    
    # make copy of orig df for later reference
    df_text_orig <- df_text
    names(df_text_orig) <- 'text_orig'
    df_text_orig$id_orig <- 1:length(df_text_orig$text_orig)
    
    # sentence tokenize
    # df_text <- tokenize_sentences(df_text$text_dm, lowercase = TRUE, strip_punct = TRUE, simplify = TRUE)
    df_text <- tokenize_sentences(df_text$text_dm, lowercase = TRUE, strip_punct = TRUE, simplify = TRUE)
    
    # convert list of sentences back into df 
    df_text <- list2df(df_text)
    
    # reaname df cols
    names(df_text) <- c('text_dm', 'id_orig')
    
    # convert id_orig to numeric
    df_text$id_orig <- as.numeric(df_text$id_orig)
    
    # add new doc id
    df_text$id_dm <- 1:length(df_text$text_dm)
    
    # join back df
    df_text <- left_join(df_text_orig, df_text,  by = "id_orig")

    # rm temp var
    rm(df_text_orig)
    
    return(df_text)
    } 
  else
    {
      return(df_text)
    }
  }

###################### ###################### ###################### text cleanup: lda modelling

func_text_preprocess <- function(df_text, func_text_preprocess=TRUE){
  
  if(func_text_preprocess){
    
    # keep copy of sentence tokennized un procesed col for sentiment analysis processing
    df_text$text_dm_orig <- df_text$text_dm
    
    # # tolower
    # df_text$text_dm <- tolower(df_text$text_dm)
    
    # drop non alpha - review step; seeing numerical values in result
    df_text$text_dm <- str_replace_all(df_text$text_dm, "[^[:alnum:]]", " ")
    
    # drop punct
    df_text$text_dm <- str_replace_all(df_text$text_dm, "[[:punct:]]", " ")
    
    # drop stop words
    # df_text$text_dm <- removeWords(df_text$text_dm, stopwords("english"))
    df_text$text_dm <- removeWords(df_text$text_dm, stopwords("en"))
    
    # lemmatize
    df_text$text_dm <- lemmatize_strings(df_text$text_dm)
    
    # drop words with nchar < 3
    df_text$text_dm <- rm_nchar_words(df_text$text_dm, "1,2")
    
    # rearrange col names for next steps
    df_text <- df_text %>% select(id_orig, text_orig, id_dm, text_dm_orig, text_dm)
    
    return(df_text)
  }
  else{
    return (df_text)
  }
}

###################### ###################### ###################### text cleanup: sentiment analysis

func_text_preprocess_sentiment <- function(df_text, func_text_preprocess=TRUE){
  
  if(func_text_preprocess){
    
    # drop non alpha - review step; seeing numerical values in result - revisit to drop numeric as well; or convert numeric to words
    df_text$text_dm_sent <- str_replace_all(df_text$text_dm_orig, "[^[:alnum:]]", " ")
    
    # drop punct
    df_text$text_dm_sent <- str_replace_all(df_text$text_dm_sent, "[[:punct:]]", " ")
    
    # drop stop words
    # df_text$text_dm_sent <- removeWords(df_text$text_dm_sent, stopwords("en"))
    
    return(df_text)
  }
  else{
    return (df_text)
  }
}

###################### ###################### ###################### ###################### ######################
###################### ###################### ###################### ###################### ######################

func_lda_master <- function(df_text, n_topics = 10){

  ###################### ###################### ###################### text pre-process
  df_text <- func_sent_tokenize(df_text)
  df_text <- func_text_preprocess(df_text)

  ###################### ###################### ###################### lda model construct
  # word tokenize
  tokens <- word_tokenizer(df_text$text_dm)

  # iterator
  it <- itoken(tokens, ids = df_text$id_dm, progressbar = FALSE)

  # create vocabulary
  v <- create_vocabulary(it) %>%
    prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.2)

  # vectorize vocab
  vectorizer <- vocab_vectorizer(v)

  # consider doing tf_idf weighting on the dtm
  # generate dtm
  dtm <- create_dtm(it, vectorizer, type = "dgTMatrix")

  lda_model <- LDA$new(n_topics = n_topics, doc_topic_prior = 0.1, topic_word_prior = 0.01)

  # fit transofrm
  doc_topic_distr <- lda_model$fit_transform(x=dtm, n_iter = 1000, convergence_tol = 0.001,
                                             n_check_convergence = 25, progressbar = FALSE)

  ###################### ###################### ###################### retag df_text by cluster id
  # combined colnames
  col_names <- c(names(df_text), paste0('topic_', 1:n_topics))

  # bind df_text with topic distrib
  df_text <- bind_cols(df_text, matrix2df(doc_topic_distr)[,-1])

  # rename dopic distrib
  colnames(df_text) <- col_names

  # reformat topic distrib to pct
  df_text[,str_detect(names(df_text), 'topic_')] <- round(df_text[,str_detect(names(df_text), 'topic_')]*100, digits = 1)

  # tag by max topic; use arg max
  df_text$main_topic <- paste0('topic_', df_text[,str_detect(names(df_text), 'topic_')] %>% max.col())

  # factor & order main topic col
  df_text$main_topic <- factor(df_text$main_topic,
                             levels = names(df_text[,str_detect(names(df_text), 'topic_')]))


  ###################### ###################### ###################### top n words df
  top_n_words <- lda_model$get_top_words(n = 10,  lambda = 1)
  colnames(top_n_words) <- paste0('topic_', 1:n_topics)

  ###################### ###################### ###################### lda vis
  lda_json <- createJSON(phi = lda_model$.__enclos_env__$private$topic_word_distribution_with_prior(),
                         theta = lda_model$.__enclos_env__$private$doc_topic_distribution_with_prior(),
                         doc.length = lda_model$.__enclos_env__$private$doc_len,
                         vocab = lda_model$.__enclos_env__$private$vocabulary,
                         term.frequency = colSums(lda_model$components),
                         lambda.step = 0.1,
                         R = 30)
  
  ###################### ###################### ###################### sentiment
  ###################### ###################### preprocess
  # preprocess text for sentiment analysis
  df_text <- func_text_preprocess_sentiment(df_text)
  
  # generate qdaps polarity scores
  df_text_pol <- polarity(text.var = df_text$text_dm_sent)
  
  # add polarity
  df_text$polarity_score <- round(df_text_pol$all$polarity, digits = 2)
  
  # add positive & negative terms
  df_text$pos_terms <- sapply(df_text_pol$all$pos.words, as.String)
  df_text$pos_terms <- str_replace_all(df_text$pos_terms, "[\r\n]" , ", ")
  
  df_text$neg_terms <- sapply(df_text_pol$all$neg.words, as.String)
  df_text$neg_terms <- str_replace_all(df_text$neg_terms, "[\r\n]" , ", ")
  
  # rm temp var
  rm(df_text_pol)
  
  ###################### ###################### jitter plot
  
  plotly_obj_polarity <- df_text %>% 
    
    plot_ly() %>% 
    add_markers(
      x = ~jitter(as.numeric(main_topic)), 
      y = ~polarity_score,
      color = ~main_topic,
      marker = list(size = 5),
      hoverinfo = "text",
      text = ~paste(
        "pol_score: ", polarity_score,
        "<br>",
        "comment: ", text_dm_sent,
        "<br>",
        "pos_terms: ", pos_terms,
        "<br>",
        "neg_terms: ", neg_terms)
    ) %>% 
    layout(
      # hovermode = 'compare',
      # title = "Topic Polarity",
      yaxis = list(title = "Polarity Score"),
      xaxis = list(title = "Topic") 
      # showticklabels = FALSE)
    ) %>% 
    layout(
      xaxis = list(
        autotick = FALSE,
        tick0 = 1,
        dtick = 1
      )
    )
  
  # configure output chart
  plotly_obj_polarity <- config(plotly_obj_polarity, displaylogo = FALSE, collaborate = FALSE)
  
  ###################### ###################### ###################### return list of outputs
  
  return(list(lda_json, top_n_words, df_text, plotly_obj_polarity))
}

###################### ###################### ###################### ###################### ######################
###################### ###################### ###################### ###################### ######################
func_lda_master_test <- function(n_topics = 10, test_df = "MS_Positive"){

  ###################### ###################### ###################### load test data
  # data("movie_review")
  # df_text <- movie_review %>% select(text_dm=review)
  
  if(test_df == "MS_Positive"){
    df_text <- ms_df_pos
  }
  if(test_df == "MS_Negative"){
    df_text <- ms_df_cons
  }
  
  if(test_df == "GS_Positive"){
    df_text <- gs_df_pos
  }
  if(test_df == "GS_Negative"){
    df_text <- gs_df_cons
  }
  # else{
  #   df_text <- NULL
  # }
  # 
  # 
  ###################### ###################### ###################### text pre-process
  df_text <- func_sent_tokenize(df_text)
  df_text <- func_text_preprocess(df_text)

  ###################### ###################### ###################### lda model construct
  # word tokenize
  tokens <- word_tokenizer(df_text$text_dm)

  # iterator
  it <- itoken(tokens, ids = df_text$id_dm, progressbar = FALSE)

  # create vocabulary
  v <- create_vocabulary(it) %>%
    prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.2)

  # vectorize vocab
  vectorizer <- vocab_vectorizer(v)

  # consider doing tf_idf weighting on the dtm
  # generate dtm
  dtm <- create_dtm(it, vectorizer, type = "dgTMatrix")

  lda_model <- LDA$new(n_topics = n_topics, doc_topic_prior = 0.1, topic_word_prior = 0.01)

  # fit transofrm
  doc_topic_distr <- lda_model$fit_transform(x=dtm, n_iter = 1000, convergence_tol = 0.001,
                                             n_check_convergence = 25, progressbar = FALSE)

  ###################### ###################### ###################### retag df_text by cluster id
  # combined colnames
  col_names <- c(names(df_text), paste0('topic_', 1:n_topics))

  # bind df_text with topic distrib
  df_text <- bind_cols(df_text, matrix2df(doc_topic_distr)[,-1])

  # rename dopic distrib
  colnames(df_text) <- col_names

  # reformat topic distrib to pct
  df_text[,str_detect(names(df_text), 'topic_')] <- round(df_text[,str_detect(names(df_text), 'topic_')]*100, digits = 1)
  
  # tag by max topic; use arg max
  df_text$main_topic <- paste0('topic_', df_text[,str_detect(names(df_text), 'topic_')] %>% max.col())

  # factor & order main topic col
  df_text$main_topic <- factor(df_text$main_topic, 
                                  levels = names(df_text[,str_detect(names(df_text), 'topic_')]))
  
  ###################### ###################### ###################### top n words df
  top_n_words <- lda_model$get_top_words(n = 10,  lambda = 1)
  colnames(top_n_words) <- paste0('topic_', 1:n_topics)

  ###################### ###################### ###################### lda vis
  lda_json <- createJSON(phi = lda_model$.__enclos_env__$private$topic_word_distribution_with_prior(),
                         theta = lda_model$.__enclos_env__$private$doc_topic_distribution_with_prior(),
                         doc.length = lda_model$.__enclos_env__$private$doc_len,
                         vocab = lda_model$.__enclos_env__$private$vocabulary,
                         term.frequency = colSums(lda_model$components),
                         lambda.step = 0.1,
                         R = 30)
  
  ###################### ###################### ###################### sentiment
  ###################### ###################### preprocess
  # preprocess text for sentiment analysis
  df_text <- func_text_preprocess_sentiment(df_text)
  
  # generate qdaps polarity scores
  df_text_pol <- polarity(text.var = df_text$text_dm_sent)
  
  # add polarity
  df_text$polarity_score <- round(df_text_pol$all$polarity, digits = 2)
  
  # add positive & negative terms
  df_text$pos_terms <- sapply(df_text_pol$all$pos.words, as.String)
  df_text$pos_terms <- str_replace_all(df_text$pos_terms, "[\r\n]" , ", ")
  
  df_text$neg_terms <- sapply(df_text_pol$all$neg.words, as.String)
  df_text$neg_terms <- str_replace_all(df_text$neg_terms, "[\r\n]" , ", ")
  
  # rm temp var
  rm(df_text_pol)
  
  ###################### ###################### jitter plot
  
  plotly_obj_polarity <- df_text %>% 
    
    plot_ly() %>% 
    add_markers(
      x = ~jitter(as.numeric(main_topic)), 
      y = ~polarity_score,
      color = ~main_topic,
      marker = list(size = 5),
      hoverinfo = "text",
      text = ~paste(
        "pol_score: ", polarity_score,
        "<br>",
        "comment: ", text_dm_sent,
        "<br>",
        "pos_terms: ", pos_terms,
        "<br>",
        "neg_terms: ", neg_terms)
    ) %>% 
    layout(
      # hovermode = 'compare',
      # title = "Topic Polarity",
      yaxis = list(title = "Polarity Score"),
      xaxis = list(title = "Topic") 
      # showticklabels = FALSE)
    ) %>% 
    layout(
      xaxis = list(
        autotick = FALSE,
        tick0 = 1,
        dtick = 1
      )
    )
  
  # configure output chart
  plotly_obj_polarity <- config(plotly_obj_polarity, displaylogo = FALSE, collaborate = FALSE)
  
  ###################### ###################### ###################### return list of outputs

  return(list(lda_json, top_n_words, df_text, plotly_obj_polarity))

}

###################### ###################### ###################### ###################### ###################### 
###################### ###################### ###################### ###################### ###################### 

# # func_sentiment_polarity <- function(df_text){
#   
#   ###################### ###################### ###################### preprocess
#   # preprocess text for sentiment analysis
#   df_text <- func_text_preprocess_sentiment(df_text)
#   
#   # generate qdaps polarity scores
#   df_text_pol <- polarity(text.var = df_text$text_dm_sent)
#   
#   # polarity sentiment data frame
#   # df_polarity <- df_text %>% select(id_dm, text_dm_sent, main_topic)
#   
#   # add polarity
#   df_text$polarity_score <- round(df_text_pol$all$polarity, digits = 2)
#   
#   # add positive & negative terms
#   # try: as.String(), list2df()
#   df_text$pos_terms <- sapply(df_text_pol$all$pos.words, as.String)
#   df_text$pos_terms <- str_replace_all(df_text$pos_terms, "[\r\n]" , ", ")
#   
#   df_text$neg_terms <- sapply(df_text_pol$all$neg.words, as.String)
#   df_text$neg_terms <- str_replace_all(df_text$neg_terms, "[\r\n]" , ", ")
#   
#   # rm temp var
#   rm(df_text_pol)
#   
#   ###################### ###################### ###################### jitter plot
#   
#   plotly_obj_polarity <- df_text %>% 
#   
#   plot_ly() %>% 
#     add_markers(
#       x = ~jitter(as.numeric(main_topic)), # update to grouping
#       y = ~polarity_score,
#       color = ~main_topic,
#       marker = list(size = 5),
#       hoverinfo = "text",
#       text = ~paste(
#         "pol_score: ", polarity_score,
#         "<br>",
#         "comment: ", text_dm_sent,
#         "<br>",
#         "pos_terms: ", pos_terms,
#         "<br>",
#         "neg_terms: ", neg_terms)
#       ) %>% 
#     layout(
#       hovermode = 'compare',
#       # title = "Topic Polarity",
#       yaxis = list(title = "Polarity Score"),
#       xaxis = list(title = "Topic") 
#                    # showticklabels = FALSE)
#            ) %>% 
#     layout(
#       xaxis = list(
#         autotick = FALSE,
#         tick0 = 1,
#         dtick = 1
#       )
#     )
#   
#   # configure output chart
#   plotly_obj_polarity <- config(plotly_obj_polarity, displaylogo = FALSE, collaborate = FALSE)
#   
#   ###################### ###################### ###################### return plotly object
#   return(plotly_obj_polarity)
#     
# # }


###################### ###################### ###################### ###################### ######################
###################### ###################### ###################### ###################### ######################


###################### ###################### ###################### ###################### ui
# ui <- dashboardPage(

# ui <- fluidPage(theme = shinytheme("slate"),

# ui <- dashboardPage(skin = 'black',

ui <- dashboardPage(
  # dashboard header
  dashboardHeader(title = h2(strong('termometer.io'))),

  # dashbaord sidebar
  dashboardSidebar(

    sidebarMenu(
      menuItem(h3("Data Source"), tabName = 'data_upload'),
      fileInput('inputFile_df_text', "Upload Your Data", accept=(".csv")),
      h5(em(" select 'Data_Upload' from dropdown")),
      h5(em(" label text field as 'text_dm'")),
      br(),br(),
      selectInput('test_data_selected', label = "Or Explore Built-in Data",
                  # choices = c("Select", "Movie Reviews"), multiple = FALSE, selected = "Select")
                  choices = c("Data_Upload", "MS_Positive", "MS_Negative", "GS_Positive", "GS_Negative"), multiple = FALSE, 
                  selected = "MS_Positive")
    ),
    br(),br(),br(),
    sidebarMenu(
      menuItem(h3("Options"), tabName = 'data_upload'),
      # sliderInput("n_topics_input", "Number of Topics", min = 3, max = 20, value = 10),
      numericInput("n_topics_input", "Number of Topics",min = 3, max = 20, step = 1, value = 8),
      br(),
      actionButton("doc_topic_distr_button", "Generate Topics", icon = icon("refresh", lib="glyphicon"))
    )
    # br(),
    # br(),
    # br()
    #
    # dropdownMenu("doc_topic_distr_button_test", "Run Test Example", icon = icon("refresh", lib="glyphicon"))

    ),

  # dashboard body
  dashboardBody(
    tabItem(tabName = "data_upload",
            
            # tabsetPanel(

              # tabPanel("Topic Modelling & Sentiment Analysis",
                       br(),
                       fluidRow(
                         box(
                           title = "Topic Modelling",
                           width = 12,
                           collapsible = TRUE,
                           visOutput('lda_plot')
                         )
                       ),
                         fluidRow(
                           box(
                             title = "Sentiment Analysis",
                             width = 12,
                             collapsible = TRUE,
                             plotlyOutput('polarity_plot')
                           )
                       ),
                       
                       fluidRow( #add download button
                         box(
                           title = "Top Terms",
                           width = 12,
                           collapsible = TRUE,
                           collapsed = TRUE,
                           dataTableOutput('top_n_words'),
                           downloadButton('download_top_n_words', 'Download Table')
                         )
                       ),
                       fluidRow( # add download button
                         box(
                           title = "All Documents",
                           width = 12,
                           collapsible = TRUE,
                           collapsed = TRUE,
                           dataTableOutput('df_text_topics_datatable'),
                           downloadButton('download_df_text_topics_datatable', 'Download Table')
                         )
                       )
                       # )

              # tabPanel("Sentiment Analysis",
              #          box(title = 'coming soon...')
              #          ),
              # 
              # tabPanel("Text Classification",
              #          box(title = 'coming soon...')
              #          )
              # )
            ),
    
    HTML(
      paste0('
      <div class="container">
             <ul class="list-inline">
             <li class="text-muted">termometer.io &copy; 2018</li>
             <li><a href="termometer.io">Privacy Policy</a></li>
             <li><a href="termometer.io">Terms of Service</a></li>
             </ul>
             </div>
             </footer>
             ')
      )
    
    )
  )
# )

###################### ###################### ###################### ###################### ui
# set max upload file size to 100mb
options(shiny.maxRequestSize = 100*1024^2)

###################### ###################### ###################### ###################### server


server <- function(input, output, session){

  # read in text data

  df_text <- reactive({
    inFile <- input$inputFile_df_text
    if(is.null(inFile)) {return (NULL)}

    ext <- tools::file_ext(inFile$name)
    file.rename(inFile$datapath, paste(inFile$datapath, ext, sep="."))
    dataFile <- read.csv(paste(inFile$datapath, ext, sep = "."), stringsAsFactors = FALSE, encoding = 'UTF-8')
    dataFile

  })

  # LDA Topics
  lda_obj_list <- eventReactive(input$doc_topic_distr_button,{

    if (input$test_data_selected == "Data_Upload") {
      return(func_lda_master(
      df_text = df_text(),
      n_topics = input$n_topics_input))}
    else{
      func_lda_master_test(
        n_topics = input$n_topics_input,
        test_df = input$test_data_selected)}
  })

  # output/render lda json
  output$lda_plot <- renderVis({
    lda_obj_list()[[1]]
  })

  # output/render polarity plot
  output$polarity_plot <- renderPlotly({
    lda_obj_list()[[4]]
  })
  
  
  # output/data table top n terms
  output$top_n_words <- renderDataTable({
    as.data.frame(lda_obj_list()[[2]])

  })

  # output/data table dtm
  output$df_text_topics_datatable <- renderDataTable({
    as.data.frame(lda_obj_list()[[3]])
  })

  # download top n terms
  output$download_top_n_words <- downloadHandler(
    filename <- function() {paste0("top_n_terms_", Sys.Date(),".csv")},
    content <- function(file) {write.csv(lda_obj_list()[[2]], file, row.names = FALSE)}
  )

  # download top n terms
  output$download_df_text_topics_datatable <- downloadHandler(
    filename <- function() {paste0("df_text_", Sys.Date(),".csv")},
    content <- function(file) {write.csv(lda_obj_list()[[3]], file, row.names = FALSE)}
  )
}

###################### ###################### ###################### ###################### run app

shinyApp(ui = ui, server = server)

# Awesome Traffic Prediction Large Models
This is a collection of research Papers about ****, and the repository will be continuously updated to track the frontier of this field. 

:clap:Welcome to follow and star! If you find any related Papers or reports could be helpful, feel free to submit an issue or PR.

## Overview
This repository presents a comprehensive review of Traffic Prediction Large Models, exploring their evolution from traditional deep learning to modern foundation models (LLMs/PFMs). We focus on multimodal traffic forecasting, where numerical sensor data integrates with real-time text for richer spatio-temporal modeling. The survey covers key methodologies, challenges, and future directions.



## Datasets
| Dataset         | Sub-dataset         | Publication                                                                                             | Frequency | Year | Data Size    | Task                     |
|-----------------|---------------------|---------------------------------------------------------------------------------------------------------|-----------|------|--------------|--------------------------|
| **libcity**     | PEMS03             | Towards efficient and comprehensive urban spatial-temporal prediction: A unified library and performance benchmark | 15 min    | 2023 | 9.38M        | Flow, Speed, Occupancy  |
|                 | PEMS04             |                                                                                                         | 5 min     | 2023 | 15.65M       | Flow, Speed, Occupancy  |
|                 | PEMS07             |                                                                                                         | 5 min     | 2023 | 24.92M       | Flow, Speed, Occupancy  |
|                 | PEMS08             |                                                                                                         | 5 min     | 2023 | 9.11M        | Flow, Speed, Occupancy  |
|                 | PEMS Bay           |                                                                                                         | 5 min     | 2023 | 16.94M       | Flow, Speed, Occupancy  |
|                 | Los-Loop           |                                                                                                         | 5 min     | 2023 | 7.09M        | Flow, Speed, Occupancy  |
|                 | Loop Seattle       |                                                                                                         | 5 min     | 2023 | 33.95M       | Flow, Speed, Occupancy  |
|                 | SZ-Taxi            |                                                                                                         | 15 min    | 2023 | 0.46M        | Flow, Speed, Occupancy  |
|                 | Beijing Subway     |                                                                                                         | 30 min    | 2023 | 0.87M        | Flow, Speed, Occupancy  |
|                 | SHMetro            |                                                                                                         | 15 min    | 2023 | 5.07M        | Flow, Speed, Occupancy  |
|                 | HZMetro            |                                                                                                         | 15 min    | 2023 | 0.38M        | Flow, Speed, Occupancy  |
|                 | Q-Traffic          |                                                                                                         | 15 min    | 2023 | 264.39M      | Flow, Speed, Occupancy  |
| **Autoformer**  | traffic            | Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting           | hourly    | 2021 | 15,122,928   | occupancy rates         |
| **Monash**      | Vehicle Trips      | Godahewa et al., Monash Time Series Forecasting Archive                                                 | hourly    | 2021 | 3.13M        | -                       |
| **largeST**     | PEMS03             | Largest: A benchmark dataset for large-scale traffic forecasting                                        | 5 min     | 2023 | -            | -                       |
|                 | PEMS04             |                                                                                                         | 5 min     | 2023 | -            | -                       |
|                 | PEMS07             |                                                                                                         | 5 min     | 2023 | -            | -                       |
|                 | PEMS08             |                                                                                                         | 5 min     | 2023 | -            | -                       |
|                 | PEMS_BAY           |                                                                                                         | 5 min     | 2023 | -            | -                       |
|                 | LOS_LOOP           |                                                                                                         | 5 min     | 2023 | -            | -                       |
|                 | LOOP_SEATTLE       |                                                                                                         | 15 min    | 2023 | -            | -                       |
|                 | SZ_TAXI            |                                                                                                         | 30 min    | 2023 | -            | -                       |
|                 | beijing_subway     |                                                                                                         | 15 min    | 2023 | -            | -                       |
|                 | SHMETROY           |                                                                                                         | 15 min    | 2023 | -            | -                       |
|                 | HZMETRO            |                                                                                                         | 2 min     | 2023 | -            | -                       |
|                 | Q-Traffic          |                                                                                                         | 15 min    | 2023 | -            | -                       |
| **gluonts**     | TAXI               | Gluonts: Probabilistic and neural time series modeling in python                                        | 30 min    | 2020 | 55.00M       | -                       |
|                 | UBER TLC DAILY     |                                                                                                         | daily     | 2020 | 0.05M        | -                       |
|                 | UBER TLC HOURLY    |                                                                                                         | hourly    | 2020 | 1.13M        | -                       |
| **TSLD**        | Traffic            | TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling                                     | hourly    | 2024 | 12,185       | -                       |
| **Istanbul-Traffic** | -              | [Kaggle Dataset](https://www.kaggle.com/datasets/leonardo00/istanbul-traffic-index)                     | 1 minute  | 2022 | 881,000      | Traffic index (congestion) |
| **TrafficText** | -                  | Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis                                      | monthly   | 2024 | 4,248        | Traffic Volume           |

## Papers
### Pure Time Series Traffic Prediction
#### PFMs  for Traffic Prediction
- [ICML2024] A decoder-only foundation model for time-series forecasting [[Paper](https://arxiv.org/abs/2310.10688)] [[Github](https://github.com/google-research/timesfm)]
- [ICML2024 Oral] Unified Training of Universal Time Series Forecasting Transformers [[Paper](https://arxiv.org/pdf/2402.02592)] [[Github](https://github.com/SalesforceAIResearch/uni2ts)]
- [ICML2024] Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series [[Paper](https://arxiv.org/pdf/2401.03955)] [[Github](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer)] [[Blog](https://aihorizonforecast.substack.com/p/tiny-time-mixersttms-powerful-zerofew)]
- [ICML2024] MOMENT: A Family of Open Time-series Foundation Models [[Paper](https://arxiv.org/pdf/2402.03885)] [[Code](https://huggingface.co/AutonLab)]
- [arXiv2024] TimeGPT-1 [[Paper](https://arxiv.org/pdf/2310.03589)] [[Github]()]
- [arXiv2024] Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting [[Paper](https://arxiv.org/pdf/2310.08278)] [[Github](https://github.com/time-series-foundation-models/lag-llama)]
- [TMLR2024] Chronos: Learning the Language of Time Series [[Paper](https://arxiv.org/pdf/2403.07815)] [[Github](https://github.com/amazon-science/chronos-forecasting)]
- [ICML2024] Timer: Generative Pre-trained Transformers Are Large Time Series Models [[Paper](https://arxiv.org/pdf/2402.02368)] [[Github](https://github.com/thuml/Large-Time-Series-Model)]
- [ICLR2025 Spotlight] Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts [[Paper](https://arxiv.org/pdf/2409.16040)] [[Github](https://github.com/Time-MoE/Time-MoE)]
- [arXiv2024] TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling [[Paper](https://arxiv.org/pdf/2402.02475)] [[Github](https://github.com/thuml/TimeSiam)]
- [NIPS2023] SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling [[Paper](https://arxiv.org/pdf/2302.00861)] [[Github](https://github.com/thuml/SimMTM)]
- [NIPS2024] TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[Paper](https://arxiv.org/pdf/2402.19072)] [[Github](https://github.com/thuml/TimeXer)]
- [ICLR2025] Timer-XL: Long-Context Transformers for Unified Time Series Forecasting [[Paper](https://openreview.net/forum?id=KMCJXjlDDr)] [[Github](https://github.com/thuml/timer-xl)]
- [NIPS2024] UNITS: A Unified Multi - Task Time Series Model [[Paper](https://arxiv.org/pdf/2403.00131)] [[Github](https://github.com/mims-harvard/UniTS)]

#### LLMs for Traffic Prediction
- [NIPS2023 Spotlight] One Fits All:Power General Time Series Analysis by Pretrained LM [[Paper](https://arxiv.org/pdf/2302.11939)] [[Github](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)]
- [NIPS2023] Large Language Models Are Zero-Shot Time Series Forecasters [[Paper](https://arxiv.org/pdf/2310.07820)] [[Github](https://github.com/ngruver/llmtime)]
- [TIST2025] LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters [[Paper](https://arxiv.org/pdf/2308.08469)] [[Github](https://github.com/liaoyuhua/LLM4TS)]
- [ICML2024] Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning [[Paper](https://arxiv.org/pdf/2402.04852)] [[Github](https://github.com/yxbian23/aLLM4TS)]
- [ICLR2024] Time-LLM: Time Series Forecasting by Reprogramming Large Language Models [[Paper](https://arxiv.org/pdf/2310.01728)] [[Github](https://github.com/KimMeen/Time-LLM)]
- [WWW2024] UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting [[Paper](https://arxiv.org/pdf/2310.09751)] [[Github](https://github.com/liuxu77/UniTime)]
- [ICLR2024] TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series [[Paper](https://openreview.net/pdf?id=Tuh4nZVb0g)] [[Github](https://github.com/SCXsunchenxi/TEST)]
- [ICLR2024] TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting [[Paper](https://arxiv.org/pdf/2310.04948)] [[Github](https://github.com/DC-research/TEMPO)]
- [ICML2024] S^2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting [[Paper](https://arxiv.org/abs/2403.05798)] [[Github](https://github.com/panzijie825/s2ip-llm)]
- [NIPS2024] AutoTimes: Autoregressive Time Series Forecasters via Large Language Models [[Paper](https://arxiv.org/pdf/2402.02370)] [[Github](https://github.com/thuml/AutoTimes)]
- [AAAI2025] CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning [[Paper](https://arxiv.org/pdf/2403.07300)] [[Github](https://github.com/Hank0626/CALF)]
- [AAAI2025] TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment [[Paper](https://arxiv.org/pdf/2406.01638)] [[Github](https://github.com/ChenxiLiu-HNU/TimeCMA)]
- [ICLR2025] Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series [[Paper](https://arxiv.org/pdf/2501.03747)] [[Github](https://github.com/tokaka22/ICLR25-FSCA)]
- [TKDE2023] PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting [[Paper](https://arxiv.org/pdf/2210.08964)] [[Github](https://github.com/HaoUNSW/PISA)]
- [SIGKDD Explorations Newsletter 2024] Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities [[Paper](https://arxiv.org/pdf/2402.10835)] [[Github](https://github.com/MingyuJ666/Time-Series-Forecasting-with-LLMs)]
- [ACL2024] LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting [[Paper](https://arxiv.org/pdf/2402.16132)] [[Github](https://github.com/AdityaLab/lstprompt)]
- [TMLR2024] LLM-TS Integrator: Integrating LLM for Enhanced Time Series Modeling [[Paper](https://arxiv.org/pdf/2410.16489)]
- [arXiv2025] Logo-LLM: Local and Global Modeling with Large Language Models for Time Series Forecasting [[Paper](https://arxiv.org/pdf/2505.11017)]

### Spatiao-temporal Traffic Prediction
### Multimodal Traffic Prediction
- [NIPS2024] Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis [[Paper](https://arxiv.org/pdf/2406.08627)] [[Github](https://github.com/AdityaLab/Time-MMD)]
- [NIPS2024] From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection [[Paper](https://arxiv.org/pdf/2409.17515)] [[Github](https://github.com/ameliawong1996/From_News_to_Forecast)]
- [AAAI2024] GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/30383)]
- [arXiv2025] Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting [[Paper](https://arxiv.org/pdf/2505.10774)]

## Talks

- [Joint Perception and Prediction for Self-driving](https://www.youtube.com/watch?v=Ce0vI_9SwNU) 
- [Joint Perception & Prediction for Autonomous Driving](https://www.youtube.com/watch?v=A-B5f9gLDh0)



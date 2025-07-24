# Awesome Traffic Prediction Large Models
This is a collection of research Papers about ****, and the repository will be continuously updated to track the frontier of this field. 

:clap:Welcome to follow and star! If you find any related Papers or reports could be helpful, feel free to submit an issue or PR.

## Overview
This repository presents a comprehensive review of Traffic Prediction Large Models, exploring their evolution from traditional deep learning to modern foundation models (LLMs/PFMs). We focus on multimodal traffic forecasting, where numerical sensor data integrates with real-time text for richer spatio-temporal modeling. The survey covers key methodologies, challenges, and future directions.



## Datasets
| Dataset         | Sub-dataset         | Publication                                                                                             | Frequency | Year | Data Size    | Task                     |
|-----------------|---------------------|---------------------------------------------------------------------------------------------------------|-----------|------|--------------|--------------------------|
| **Libcity**     | PEMS03             | [LibCity: A Unified Library Towards Efficient and Comprehensive Urban Spatial-Temporal Prediction](https://arxiv.org/pdf/2304.14343) | 5 min    | 2023 | 9.38M        | Flow  |
|                 | PEMS04             |                                                                                                         | 5 min     | 2023 | 15.65M       | Flow, Speed  |
|                 | PEMS07             |                                                                                                         | 5 min     | 2023 | 24.92M       | Flow  |
|                 | PEMS08             |                                                                                                         | 5 min     | 2023 | 9.11M        | Flow, Speed  |
|                 | PEMS Bay           |                                                                                                         | 5 min     | 2023 | 16.94M       | Speed  |
|                 | Los-Loop           |                                                                                                         | 5 min     | 2023 | 7.09M        | Speed  |
|                 | Loop Seattle       |                                                                                                         | 15 min     | 2023 | 33.95M       | Speed  |
|                 | SZ-Taxi            |                                                                                                         | 30 min    | 2023 | 0.46M        | Speed  |
|                 | Beijing Subway     |                                                                                                         | 15 min    | 2023 | 0.87M        | Flow, Demand  |
|                 | SHMetro            |                                                                                                         | 15 min    | 2023 | 5.07M        | Flow, Demand  |
|                 | HZMetro            |                                                                                                         | 2 min    | 2023 | 0.38M        | Flow, Demand  |
|                 | Q-Traffic          |                                                                                                         | 15 min    | 2023 | 264.39M      | Speed  |
| **Autoformer**  | Traffic            | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/pdf/2106.13008)           | Hourly    | 2021 | 15M   | occupancy rates         |
| **Monash**      | Vehicle Trips      | [Monash Time Series Forecasting Archive](https://arxiv.org/pdf/2105.06643)     | Daily    | 2021 | 3.13M        | -                       |
|  | Pedestrian Counts |  | Hourly | 2021 |  | hourly pedestrian counts |
|  | rideshare |  | Hourly | 2021 |  | attributes related to Uber and Lyft rideshare services |
| **LargeST**     | PEMS03 PEMS04 PEMS07 PEMS08 PEMS_BAY LOS_LOOP LOOP_SEATTLE SZ_TAXI Beijing_subway SHMETROY  HZMETRO Q-Traffic            | [LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting](https://arxiv.org/pdf/2306.08259)                                    | 2 min-30 min     | 2023 | -            | Flow, Speed, Demand                       |         
| **Gluonts**     | TAXI               | [Gluonts: Probabilistic and neural time series modeling in python](https://arxiv.org/pdf/1906.05264)                                        | 30 min    | 2020 | 55.00M       | -                       |
|    | UBER TLC DAILY     |  | Daily     | 2020 | 0.05M        | Uber pickups                       |
|                 | UBER TLC HOURLY    |                 | Hourly    | 2020 | 1.13M        | Uber pickups                       |
| **Istanbul-Traffic** | Istanbul-Traffic              | [Kaggle Dataset](https://www.kaggle.com/datasets/leonardo00/istanbul-traffic-index)                     | 1 min  | 2022 | 0.88M      | Traffic index (congestion) |
| **TrafficText** | TrafficText                  | [Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis](https://arxiv.org/pdf/2406.08627)                                      | Monthly   | 2024 | 0.004M        | Traffic Volume           |

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
- [NeurIPS2023] SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling [[Paper](https://arxiv.org/pdf/2302.00861)] [[Github](https://github.com/thuml/SimMTM)]
- [NeurIPS2024] TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[Paper](https://arxiv.org/pdf/2402.19072)] [[Github](https://github.com/thuml/TimeXer)]
- [ICLR2025] Timer-XL: Long-Context Transformers for Unified Time Series Forecasting [[Paper](https://openreview.net/forum?id=KMCJXjlDDr)] [[Github](https://github.com/thuml/timer-xl)]
- [NeurIPS2024] UNITS: A Unified Multi - Task Time Series Model [[Paper](https://arxiv.org/pdf/2403.00131)] [[Github](https://github.com/mims-harvard/UniTS)]

#### LLMs for Traffic Prediction
- [NeurIPS2023 Spotlight] One Fits All:Power General Time Series Analysis by Pretrained LM [[Paper](https://arxiv.org/pdf/2302.11939)] [[Github](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)]
- [NeurIPS2023] Large Language Models Are Zero-Shot Time Series Forecasters [[Paper](https://arxiv.org/pdf/2310.07820)] [[Github](https://github.com/ngruver/llmtime)]
- [TIST2025] LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters [[Paper](https://arxiv.org/pdf/2308.08469)] [[Github](https://github.com/liaoyuhua/LLM4TS)]
- [ICML2024] Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning [[Paper](https://arxiv.org/pdf/2402.04852)] [[Github](https://github.com/yxbian23/aLLM4TS)]
- [ICLR2024] Time-LLM: Time Series Forecasting by Reprogramming Large Language Models [[Paper](https://arxiv.org/pdf/2310.01728)] [[Github](https://github.com/KimMeen/Time-LLM)]
- [WWW2024] UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting [[Paper](https://arxiv.org/pdf/2310.09751)] [[Github](https://github.com/liuxu77/UniTime)]
- [ICLR2024] TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series [[Paper](https://openreview.net/pdf?id=Tuh4nZVb0g)] [[Github](https://github.com/SCXsunchenxi/TEST)]
- [ICLR2024] TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting [[Paper](https://arxiv.org/pdf/2310.04948)] [[Github](https://github.com/DC-research/TEMPO)]
- [ICML2024] S^2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting [[Paper](https://arxiv.org/abs/2403.05798)] [[Github](https://github.com/panzijie825/s2ip-llm)]
- [NeurIPS2024] AutoTimes: Autoregressive Time Series Forecasters via Large Language Models [[Paper](https://arxiv.org/pdf/2402.02370)] [[Github](https://github.com/thuml/AutoTimes)]
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
- [NeurIPS2024] Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis [[Paper](https://arxiv.org/pdf/2406.08627)] [[Github](https://github.com/AdityaLab/Time-MMD)]
- [NeurIPS2024] From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection [[Paper](https://arxiv.org/pdf/2409.17515)] [[Github](https://github.com/ameliawong1996/From_News_to_Forecast)]
- [AAAI2024] GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/30383)]
- [arXiv2025] Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting [[Paper](https://arxiv.org/pdf/2505.10774)]

## Talks

- [Joint Perception and Prediction for Self-driving](https://www.youtube.com/watch?v=Ce0vI_9SwNU) 
- [Joint Perception & Prediction for Autonomous Driving](https://www.youtube.com/watch?v=A-B5f9gLDh0)



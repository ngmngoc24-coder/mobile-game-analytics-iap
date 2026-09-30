## Mobile Game In-App Purchase Dataset

## 1. Dataset Overview
This dataset contains player-level information for mobile games. It combines player demographics, platform information, engagement metrics, monetization value, and purchase behavior. It is suitable for:
- player segmentation 
- monetization analysis
- NOT standard retention or lifecycle analysis
The dataset contains 3,024 player records and 13 variables.

## 2. Data Grain

### Unit of Observation
One row represents one player.
Each player is uniquely identified by `UserID`.
Therefore, the dataset is structured at the player level rather than the session level or transaction level.

### Unique Values
UserID                           3024
Age                                42
Gender                              3
Country                            27
Device                              2
GameGenre                          15
SessionCount                       21
AverageSessionLength             1915
SpendingSegment                     3
InAppPurchaseAmount              1839
FirstPurchaseDaysAfterInstall      31
PaymentMethod                       7
LastPurchaseDate                  225

### Data Structure
The variables can be grouped into six analytical categories:
1. Player Attributes
2. Platform
3. Game Context
4. Engagement
5. Monetization
6. Purchase Behavior


#### 1. Player Attributes

- `UserID`
- `Age`
- `Gender`
- `Country`

#### 2. Platform

- `Device`

#### 3. Game Context

- `GameGenre`

#### 4. Engagement

- `SessionCount`
- `AverageSessionLength`

#### 5. Monetization

- `SpendingSegment`
- `InAppPurchaseAmount`

#### 6. Purchase Behavior

- `FirstPurchaseDaysAfterInstall`
- `PaymentMethod`
- `LastPurchaseDate`

## 3. Variable Dictionary

| Variable | Data Type | Category | Description |
|---|---|---|---|
| `UserID` | Identifier | Player Attribute | Unique identifier for each player |
| `Age` | Numeric | Player Attribute | Age of the player |
| `Gender` | Categorical | Player Attribute | Gender of the player |
| `Country` | Categorical | Player Attribute | Country associated with the player |
| `Device` | Categorical | Platform | Device used by the player |
| `GameGenre` | Categorical | Game Context | Genre of the game |
| `SessionCount` | Integer | Engagement | Number of sessions recorded for the player |
| `AverageSessionLength` | Numeric | Engagement | Average duration of a player's sessions |
| `SpendingSegment` | Categorical | Monetization | Player spending classification |
| `InAppPurchaseAmount` | Numeric | Monetization | Recorded in-app purchase amount |
| `FirstPurchaseDaysAfterInstall` | Numeric | Purchase Behavior | Number of days between installation and first purchase |
| `PaymentMethod` | Categorical | Purchase Behavior | Payment method used for purchases |
| `LastPurchaseDate` | Date | Purchase Behavior | Date of the player's last recorded purchase |

## 4. Analytical Scope
The dataset can be used to investigate the following areas:

### Player Composition

- Player demographics
- Geographic distribution
- Device distribution
- Game genre distribution

### Engagement

- Session frequency
- Average session duration
- Differences in engagement across player segments

### Monetization

- Revenue distribution
- Spending segment composition
- ARPU and ARPPU
- Revenue concentration
- Monetization differences across player groups

### Purchase Behavior

- Time to first purchase
- Payment method distribution
- Relationship between purchase timing and spending

### Player Segmentation

- Comparison of Minnow, Dolphin, and Whale players
- Identification of behavioral characteristics associated with high-value players

### Predictive Analysis

- Predicting payer status
- Identifying characteristics associated with high-value players

## 5. Analytical Scope & Limitations

### Player-Level Dataset

The dataset is structured at the player level. It does not contain individual session, purchase transaction, or gameplay event records.

### Retention Analysis

The dataset does not contain sufficient event-level activity history to calculate standard retention metrics such as D1, D7, or D30 retention reliably.

### DAU / WAU / MAU

Daily, weekly, and monthly active-user metrics cannot be calculated reliably because detailed player activity timestamps are not available.

### Churn

`LastPurchaseDate` should not be interpreted as `LastActiveDate`. Therefore, purchase inactivity cannot automatically be classified as player churn.

### Causality

Relationships identified between engagement and monetization should be interpreted as associations rather than causal effects. The dataset does not provide an experimental design that would allow causal conclusions.

### Data Required for Future Analysis

To conduct deeper lifecycle, retention, and game economy analysis, the project would benefit from event-level data including:

- Player installation timestamp
- Session start and end timestamps
- Login/activity events
- Purchase transaction timestamps
- Purchase item information
- Gameplay events
- Level progression
- In-game currency transactions
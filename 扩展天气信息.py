店铺对应州：
	分隔符为,。
	字段：
		Store，State。
	用于连接每家店铺与州。
天气：
	分隔符为;，注意一下。
	字段：
		"Date";
		"Max_TemperatureC";"Mean_TemperatureC";"Min_TemperatureC";
		"Dew_PointC";"MeanDew_PointC";"Min_DewpointC";
		"Max_Humidity";"Mean_Humidity";"Min_Humidity";
		"Max_Sea_Level_PressurehPa";"Mean_Sea_Level_PressurehPa";"Min_Sea_Level_PressurehPa";
		"Max_VisibilityKm";"Mean_VisibilityKm";"Min_VisibilitykM";
		"Max_Wind_SpeedKm_h";"Mean_Wind_SpeedKm_h";
		"Max_Gust_SpeedKm_h";
		"Precipitationmm"; - a lot of NA
		"CloudCover";
		"Events";
		"WindDirDegrees"
	直观感觉：
		Date用于对应其他表，温度、湿度、可见距离（涉及到开车，而外国人开车很普遍）、风速（是否愿意出门）、Events（是否适合出门）。
		Precipitationmm基本都是NA，抛弃掉
		是否需要结合之前一段时间的天气（温度、风速），毕竟不一样的温度和风度、湿度可能导致各种疾病的爆发，也会影响之后的相关药品的销售。
		
'''
store_states.csv

Brandenburg.csv (82.24 KB)
Sachsen.csv (82.76 KB)
Berlin.csv (82.67 KB)
NordrheinWestfalen.csv (82.71 KB)
SchleswigHolstein.csv (81.35 KB)
Niedersachsen.csv (82.77 KB)
Thüringen.csv (83.13 KB)
MecklenburgVorpommern.csv (81.35 KB)
SachsenAnhalt.csv (80.04 KB)
RheinlandPfalz.csv (81.1 KB)
Hamburg.csv (83.02 KB)
Saarland.csv (82.22 KB)
Bremen.csv (83.16 KB)
BadenWürttemberg.csv (82.57 KB)
Bayern.csv (81.58 KB)
Hessen.csv (82.69 KB)
'''

'''
以下是德国16个联邦州（das Bundesland -"er)标准缩写：
BB Brandenburg 勃兰登堡
BE Berlin 柏林
BW Baden-Würtemberg 巴登-符腾堡
BY Bayern 拜仁
HB Bremen 不莱梅
HE Hessen 黑森
HH Hamburg 汉堡
MV Mecklenburg-Vorpommern 梅克伦堡-前波美尼亚
NI Niedersachsen 下萨克森
NRW Nordrhein-Westfalen 北莱茵-威斯特法伦
RP Rheinland-Pfalz 莱茵兰-普法尔茨
SE Sachsen 萨克森
SH Schleswig-Holstein 石勒苏益格-荷尔斯泰因
SL Saarland 萨尔
ST Sachsen-Anhalt 萨克森-安哈尔特
TH Thüringen 图林根
'''

'''
HE-Hessen, TH-Thüringen, NW-Nordrhein-Westfalen, BE-Berlin, SN-Sachsen, SH-Schleswig-Holstein, "HB,NI"-Niedersachsen?, BY-Bayern, BW-Baden-Würtemberg, RP-Rheinland-Pfalz, ST-Sachsen-Anhalt, HH-Hamburg
'''

# 拼接所有天气数据
berlinData = pd.read_csv('./data/weather/Berlin.csv', req=';')
berlinData.State='BE'
hanboData = pd.read_csv('./data/weather/Hanbo.csv', req=',')
hanboData.State='HB'
....
weatherData = pd.concat(berlinData, hanboData)
....
# Precipitationmm字段NA太多
weatherData.drop(['Precipitationmm'], axis=1, inplace=True)

# 映射商店到某个州
storeStatesData = pd.read_csv('./data/store_states.csv')
trainDataWithState = pd.merge(trainData, storeStatesData, on='Store')

# 通过州、日期合并天气数据
trainDataWithWeatherAndState = pd.merge(trainDataWithState, weatherData, on=['Date','State'])

# 看下各州的销售情况
trainDataWithWeatherAndState.groupby(['State'])['Sales'].plot()

# 看下天气跟销售的关系
trainDataWithWeatherAndState.groupby(['Events'])['Sales'].plot()

import requests

from bot_core.config import get_secret


def get_weather(city, forecast=False):
    """
    查询指定城市的天气信息

    Args:
        city (str): 城市名称
        forecast (bool): 是否查询天气预报（默认False为实况天气）

    Returns:
        dict: 天气信息的JSON数据，如果出错则返回None
    """
    apiKey = get_secret("amap_api_key", "")
    if not apiKey:
        raise ValueError("未配置高德天气 API Key，请在 app_secrets.json 中设置 amap_api_key")

    if forecast:
        # 天气预报接口
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={apiKey}&city={city}&extensions=all"
    else:
        # 实况天气接口
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?key={apiKey}&city={city}"

    print(f"请求URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求出错: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


# 测试调用
if __name__ == "__main__":
    # 测试实况天气
    weather_data = get_weather("南宁")
    if weather_data:
        print("实况天气信息:", weather_data)

    # 测试天气预报
    forecast_data = get_weather("南宁", forecast=True)
    if forecast_data:
        print("天气预报信息:", forecast_data)

import mysql.connector
import os, json, time, re, math
import pandas as pd
import numpy as np
import tiktoken
from openai import AzureOpenAI
import json
from collections import defaultdict


# SELECT 문 실행. 
# - 에러 발생 시, 에러문 str return
# - 그렇지 않으면, pandas.DataFrame 으로 return
def select_table(mysql_config, query):
    conn = mysql.connector.connect(
            host = mysql_config['host'],          
            user = mysql_config['user'],      
            password = mysql_config['password'],
            database = mysql_config['database'],
            ssl_ca=mysql_config['ssl_ca'], 
            ssl_disabled=False)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns = col_names)
        return df
    
    except Exception as e :
        return f"{str(e)}"
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# INSERT시 사용
# - 에러 발생시, 에러문 str return      
def commit_query_w_vals(mysql_config, query, vals, many_tf=False):
    conn = mysql.connector.connect(
            host = mysql_config['host'],          
            user = mysql_config['user'],      
            password = mysql_config['password'],
            database = mysql_config['database'],
            ssl_ca=mysql_config['ssl_ca'], 
            ssl_disabled=False)
    try:
        cursor = conn.cursor()
        if many_tf:
            cursor.executemany(query, vals)
        else :
            cursor.execute(query, vals)
        conn.commit()
        print(f"INSERT - {cursor.rowcount} record(s)")
    except Exception as e :
        print("Error while connecting to MySQL", str(e))
        raise
        
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# DELETE, UPDATE 시 사용
def commit_query(mysql_config, query):
    conn = mysql.connector.connect(
            host = mysql_config['host'],          
            user = mysql_config['user'],      
            password = mysql_config['password'],
            database = mysql_config['database'],
            ssl_ca=mysql_config['ssl_ca'], 
            ssl_disabled=False)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        print(f"DELETE/UPDATE - {cursor.rowcount} record(s)")
        
    except Exception as e :
        print("Error while connecting to MySQL", str(e))
        
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            
# ==================================== GPT 관련 ====================================

# 토큰 수 계산 함수 정의
def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
    # logger.info("###### Function Name : num_tokens_from_string")
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def extract_page_patterns(input_string:str):
    ## 정규 표현식을 사용해 "<page:x>" 형태의 패턴을 찾음.
    pattern = re.compile(r'<page:\d+>')
    matches = pattern.findall(input_string)
    return matches


def split_text_by_batch(full_text, num_pages, page_batch_size = 40):
    # GPT 실행용 텍스트 분할 리스트 생성 : page_batch_size 단위로
    data_list = []
    if (num_pages < page_batch_size + 10) & (num_tokens_from_string(full_text) <= 120000):
        # batch 크기 + 10 보다  페이지 수가 작고, 전체 토큰수가 120K 이하이면
        #   전체 텍스트를 사용
        data_list.append({"start_page": 1
                          , "end_page": num_pages
                          , "data" : full_text
                        })
        
    elif (num_pages < page_batch_size + 10) & (num_tokens_from_string(full_text) > 120000):
        # batch 크기 + 10 보다  페이지 수가 작고, 전체 토큰수가 120K 이상이면
        #   절반으로 나누기
        for i in range(2):
            st_page_num = int(num_pages/2 * i + 1)
            ed_page_num = int(num_pages/2 * (i+1) + 1)
            if ed_page_num > num_pages:
                ed_page_num = None
                data_list.append({"start_page": st_page_num
                                 , "end_page": num_pages
                                 , "data" :full_text[
                                                full_text.find(f"<page:{st_page_num}>"):
                                            ]
                                })
                
            else:
                data_list.append({"start_page": st_page_num
                                 , "end_page": ed_page_num-1
                                 , "data" :full_text[
                                                full_text.find(f"<page:{st_page_num}>"):
                                                full_text.find(f"<page:{ed_page_num}>")
                                            ]
                                })

    else :
        # 그 외
        ## 읽어온 파일을 page_batch_size 페이지 단위로 분할
        for idx in range(math.ceil(num_pages/page_batch_size)):
            st_page_num = page_batch_size * idx + 1
            ed_page_num = page_batch_size * (idx+1) + 1
            
            if ed_page_num > num_pages:
                ed_page_num = None
                sliced_text = full_text[
                    full_text.find(f"<page:{st_page_num}>"):
                ]
            else:
                sliced_text = full_text[
                    full_text.find(f"<page:{st_page_num}>"):
                    full_text.find(f"<page:{ed_page_num}>")
                ]

            # 토큰수 확인
            if (num_tokens_from_string(sliced_text) <= 120000) & (ed_page_num != None):
                data_list.append({"start_page": st_page_num
                                , "end_page": ed_page_num-1
                                , "data" : sliced_text
                                })
                
            elif (num_tokens_from_string(sliced_text) <= 120000) & (ed_page_num == None):
                data_list.append({"start_page": st_page_num
                                , "end_page": num_pages
                                , "data" : sliced_text
                                })
                
            elif (ed_page_num == None):
                # 토큰수가 120K 초과이고, ed_page_num이 none인경우
                half_page_num = int(st_page_num + (num_pages-st_page_num)/2)
                data_list.append({"start_page": st_page_num
                                , "end_page": half_page_num-1
                                , "data" : sliced_text[
                                            sliced_text.find(f"<page:{st_page_num}>"):
                                            sliced_text.find(f"<page:{half_page_num}>")
                                        ]
                            })
                
                data_list.append({"start_page": half_page_num
                                , "end_page": num_pages
                                , "data" : sliced_text[
                                            sliced_text.find(f"<page:{half_page_num}>"):
                                        ]
                            })
            
            else : 
                # 토큰수가 120K 초과이고, ed_page_num이 none이 아닌 경우
                half_page_num = int(st_page_num + page_batch_size/2)
                data_list.append({"start_page": st_page_num
                                , "end_page": half_page_num-1
                                , "data" : sliced_text[
                                            sliced_text.find(f"<page:{st_page_num}>"):
                                            sliced_text.find(f"<page:{half_page_num}>")
                                        ]
                            })
                
                data_list.append({"start_page": half_page_num
                                , "end_page": ed_page_num-1
                                , "data" : sliced_text[
                                            sliced_text.find(f"<page:{half_page_num}>"):
                                            sliced_text.find(f"<page:{ed_page_num}>")
                                        ]
                            })
    return data_list


def get_system_mssg(pcd, section_group_df):
    ## 제품군 또는 제품에 따른 섹션 그룹 정보 매핑 & 섹션 분할 시스템 프롬프트 생성
    groups = section_group_df[section_group_df['PRODUCT_LVL1_CD']==pcd]['GPT_GROUP_VAL'].to_list()
    if len(groups) == 0 :
        return "Wrong Product!"
    groups.append("그외") # 지정된 그룹에 할당할수없는 내용인 경우 처리
    
    rule_txt = """As an insightful customer support analyst for LG Electronics, your task is to classify pages of the given <Documents>.\nThe <Documents> are Electronic Service Technical Manual for LG Electronics products and are composed in Korean.\nYou Must adhere to <Rules>.\n\n<Rules>\n- You MUST classify <Documents> into only ONE category per page.\n- If the content of a page corresponds to several categories, MUST classify it into only ONE Category with the most relevant.\n- There are description of <Categories> is as follows."""
    
    category_txt = "<Categories>\n"
    for ind, g in enumerate(groups):
        category_txt = category_txt + f"{ind+1}. {g}\n"
    
    output_txt = """The output should be of the following form\n```text\n- Page {number}: {category}\n- Page {number}: {category}\n- Page {number}: {category}\n```"""
    
    return f"{rule_txt}\n{category_txt}\n{output_txt}", groups


def parse_text_to_dict(input_text:str):
    # -- 생성된 텍스트를 딕셔너리로 변환
    # 결과를 저장할 딕셔너리
    result_dict = {}
    # 입력 텍스트를 줄 단위로 분할
    lines = input_text.split("\n")
    
    for line in lines:
        # 각 줄에서 페이지 번호와 텍스트 분리
        if ": " in line:
            # ":" 기호를 기준으로 분리하여 페이지 번호와 텍스트를 추출
            page_info, text = line.split(": ")
            # 페이지 번호에서 숫자만 추출
            page_number = int(page_info.split(" ")[-1])
            # 딕셔너리에 텍스트를 키로 하여 페이지 번호 추가
            if text in result_dict:
                result_dict[text].append(page_number)
            else:
                result_dict[text] = [page_number]        
    return result_dict


def merge_dicts(dict1, dict2):
    # -- 두 개의 딕셔너리 병합
    # 결과 딕셔너리 초기화
    result = {}

    # 두 딕셔너리의 키 합집합을 순회
    for key in set(dict1) | set(dict2):
        if key in dict1 and key in dict2:
            # 리스트의 경우 합치고 정렬
            if isinstance(dict1[key], list) and isinstance(dict2[key], list):
                result[key] = sorted(dict1[key] + dict2[key])
            # 딕셔너리의 경우 재귀적으로 통합
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                result[key] = merge_dicts(dict1[key], dict2[key])
            else:
                # 같은 키가 있지만 타입이 다른 경우 리스트로 담아 통합
                result[key] = sorted(list(dict1[key]) + list(dict2[key]))
        elif key in dict1:
            result[key] = dict1[key]
        else:
            result[key] = dict2[key]
    return result

# [2024.05.22] endpoint, key 등을 가져오는 방식 변경
def get_response(system_message:str, user_message:str, config:dict, max_tokens:int=4096):

    client = AzureOpenAI(
        azure_endpoint = config['account_endpoint'], 
        api_key = config['account_key'],
        api_version = config['api_version'],
    )
    model = config['model_name']

    messages = [{
            "role":"system",
            "content":system_message
        },{
            "role":"user",
            "content":user_message
        }]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    # response = request_AOAI(client=client, messages=messages, model=model, max_tokens=max_tokens)
    completion_message = response.choices[0].message.content

    return response, completion_message 


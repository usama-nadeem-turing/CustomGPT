WITH 
  base as (select a.developer_id,  a.taap_score, b.created_date
from `turing-230020.raw.ms2_job_match_pre_shortlist` a
left join `turing-230020.raw.tpm_dev_grading_result` b on a.developer_id = b.developer_id and transcript_url like "%qode%"
where a.job_id IN( 
  52768,
  61743,
  61744
)
),
devs AS (
  SELECT
    developer_id,
    years_of_experience AS YoE,
    concat("https://matching.turing.com/developer/", developer_id) as matching_link,
    concat("https://storage.googleapis.com/turing_developers/resume/", resume) as resume_link,
    status as vetting_status,
    resume_plain,
    os.phase,
    country_name


  FROM `turing-230020.sot.developer` sot
  LEFT JOIN `turing-230020.custom_dashboards.Ordered_Status` os USING(status)
  LEFT JOIN `turing-230020.raw.tpm_countries` tc on sot.country_id = tc.id
  WHERE 
    is_cheater = 0
    AND os.phase IN ("1. Phase 1", "2. Phase 2")
    AND resume IS NOT NULL
    AND resume_plain IS NOT NULL
    AND resume_plain <> ""
    /*AND developer_id NOT IN 
    (
        SELECT distinct 
          developer_id,
        FROM `turing-230020.raw.ms2_job_match_pre_shortlist` 
        WHERE 
          job_id IN (52768)
    )*/


    AND IFNULL(tc.is_OFAC, 0) = 0

    AND COALESCE(years_of_experience, calculated_years_of_experience) between 0 AND 99
    -- AND country_name = 'United States'


    -- AND developer_id = 1922725
)




SELECT distinct 
  developer_id,
  -- YoE,
  country_name,
  -- phase,
  devs.resume_plain,
  -- matching_link,
  -- resume_link,
  -- vetting_status,

FROM devs


WHERE 

  developer_id IN (


      select distinct developer_id from base 
      -- where taap_score =1 or taap_score is null or created_date is not null
  )
  /*
  AND developer_id NOT IN (
    select distinct developer_id FROM `turing-dev-337819.custom_dashboards.swe_he_v4_sheet`
  )
  */
  AND developer_id NOT IN (
    6571686,
    6596846
  )

LIMIT 30
  

ORDER BY 2 desc




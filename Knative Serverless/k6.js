import http from 'k6/http';
import { check } from 'k6';


export const options = {
  scenarios: {
    rampa_lineare_continua: {
      executor: 'ramping-arrival-rate',
      startRate: 0,          
      timeUnit: '1s',
      preAllocatedVUs: 100,   
      maxVUs: 15000,           
      stages: [
        
        { target: 900, duration: '3m' },
       
        { target: 0, duration: '2s' },
      ],
    },
  },
};
const payload = JSON.stringify({
  instances: [
    [1.5, 2.3, 3.1, 4.0, 5.5, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
  ]
});

const params = {
  headers: {
    'host': 'breast-cancer-gradient-boosting.kserve-inference.svc.cluster.local',
    'Content-Type': 'application/json',
  },
  timeout: '10s', 
};

export default function () {
  const url = 'http://breast-cancer-gradient-boosting-predictor.kserve-inference.svc.cluster.local/v1/models/breast-cancer-gradient-boosting:predict';
  
  const res = http.post(url, payload, params);
  
  const success = check(res, {
    'status is 200': (r) => r.status === 200,
  });

  
  if (!success) {
    console.log(`Status: ${res.status} | Body: ${res.body}`);
  }
}

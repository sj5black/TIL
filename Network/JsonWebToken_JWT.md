## **02. JSON Web Token (JWT)**

### ✔️ JWT란 무엇인가요?

JWT는 다양한 장치에서 공통적으로 사용할 수 있는 **Token 기반 인증 방식** 중 하나로, 인증 데이터를 담은 토큰입니다.  
토큰 자체에 유저에 대한 간단한 정보가 포함되어 있으며, 이를 통해 인증을 처리합니다.

---

### Session & Cookie

#### ⭐ 쿠키 (Cookie)
- 웹 브라우저와 요청 및 응답 시 사용하는 데이터 조각
- 특정 도메인에 제한적이며 유효기간이 정해져 있음
- 인증(Auth) 외에도 다양한 방식으로 활용 가능

#### ⭐ 세션 (Session)
- **Stateless한 HTTP**의 특징을 보완하기 위한 방법
- 세션 DB를 이용해 유저 정보를 기억하고, **Session ID**를 쿠키에 담아 인증에 활용
- 쿠키를 사용해 세션 ID를 주고받는 방식

---

### JSON Web Token (JWT)

#### ✅ 간단 개요
- 쿠키는 웹 브라우저에서만 사용되지만, JWT는 다양한 장치에서 사용 가능
- JWT는 **랜덤한 문자열**로, 간단한 서명을 포함하며 유저 정보를 담고 있음
- JWT로 인증을 처리하면 세션 DB나 복잡한 인증 로직이 필요 없음

---

### JWT 인증 처리 방식

1. 클라이언트가 **ID/PW**를 서버로 전송
2. 서버는 **ID/PW**를 검증한 후, 유효하다면 서명 처리된 토큰을 클라이언트에 응답
3. 클라이언트는 **모든 요청 헤더에 토큰**을 포함해 서버로 전송
4. 서버는 토큰의 유효성을 검증한 뒤, 유저 신원과 권한을 확인해 요청을 처리

---

### 세션과 JWT의 차이점

| 구분   | 세션 방식                              | JWT 방식                              |
|--------|---------------------------------------|---------------------------------------|
| 데이터 | 세션 DB 필요                          | 토큰 자체에 인증 데이터 포함          |
| 처리   | 세션 DB에 저장된 데이터 확인          | 토큰 유효성만 검증                    |
| 확장성 | 상태 기반 (Stateful)                  | 무상태 기반 (Stateless)               |

---

### JWT의 장점과 단점

#### 🙆‍♂️ 장점
- 서버에서 별도의 데이터를 관리하지 않아 복잡한 처리 로직이 필요 없음
- 세션이나 DB 없이 유저 인증 가능

#### 💁‍♂️ 단점
- **로그아웃** 등 세션 관리가 어렵고, 모든 기기에서 로그아웃 처리 불가능
- 토큰이 탈취되면 보안에 취약

---

### JWT 구조

#### JWT는 `.`을 기준으로 **HEADER**, **PAYLOAD**, **SIGNATURE** 세 부분으로 구성됩니다.

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9. eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ. SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
```

#### 1. **HEADER**
- 토큰 타입 (예: JWT)과 서명 생성 알고리즘(예: HS256) 정보 포함

#### 2. **PAYLOAD**
- 토큰 발급자, 대상자, 만료 시간 등 여러 데이터 포함
- 유저의 최소한의 정보 저장 (예: 유저 ID 또는 PK)
- 민감한 정보는 담지 않음 (누구나 디코딩 가능)

#### 3. **SIGNATURE**
- `HEADER` + `PAYLOAD` + **비밀키**로 생성된 서명
- **유효성 검증**: 토큰이 위변조되지 않았는지 확인
- 서버는 토큰의 서명을 검증하여 유효한 요청인지 판단

---

### Access Token과 Refresh Token

#### 🔑 문제: 토큰 탈취 시 보안 문제
JWT 인증은 장점이 많지만, 탈취 시 보안에 취약합니다. 이를 해결하기 위해 **토큰 유효시간을 짧게 설정**하고, 두 종류의 토큰을 사용합니다.

#### ➡️ **Access Token**
- 인증 요청 시 헤더에 포함되는 토큰
- 만료 기한을 짧게 설정 (탈취 시 피해 최소화)

#### 🔃 **Refresh Token**
- Access Token이 만료되었을 때 새로운 Access Token을 발급받기 위한 토큰
- 더 긴 유효기간을 가짐
- 주로 클라이언트(기기)에 저장
- Refresh Token까지 만료되면 재인증 필요

#### Refresh Token 보안 강화
- Refresh Token은 DB를 이용해 관리 가능
  - 예: Blacklist 방식으로 탈취 방지

---

### 참고 사이트
- [JWT 공식 사이트](https://jwt.io/)
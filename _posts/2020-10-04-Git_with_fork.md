---
title:  "Git_with_Fork"
excerpt: "Fork 사용법"
toc: true
toc_sticky: true
categories:
  - Git
tags:
  - Git
last_modified_at: 2020-10-04
---

# Fork

* original 프로젝트를 copy해서 내 repository로 가져온다
* copy 프로젝트에서 git remote add origin (repository주소)후 작업을 한다
* copy repository에 push한다
* original repository로 copy repositoey에서 pull requests를 한다

# Fork repository 최신 버전 유지

* 등록 소스확인

```
git remote -v
origin	(copy repository 주소) (fetch)
origin	(copy repository 주소) (push)
```

* 원본 소스 등록

```
git remote add upstream (original repository 주소)
```

```
git remote -v
origin	(copy repository 주소) (fetch)
origin	(copy repository 주소) (push)
upstream  (original repository 주소) (fetch)
upstream  (original repository 주소) (push)
```

* 잘못 등록했을경우

```
git remote remove upstream
```

* 동기화 local로 받기

```
git fetch upstream
```

* copy repository에 merge

```
git merge upstream/master
```

* add, commit, push 

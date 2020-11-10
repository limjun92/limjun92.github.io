---
title:  "git commit 되돌리기"
excerpt: "git commit 되돌리기"
toc: true
toc_sticky: true
categories:
  - Git
tags:
  - Git
last_modified_at: 2020-11-11
---

# 삭제한 commit 살리기

## commit 확인

```
git reflog
```

## 삭제하기

```
git reset --hard HEAD~4
or 
git reset --hard e26d1b0
```

## 되돌리기

```
git reflog
```

* 확인

## 복원 커밋으로 체크아웃

```
git checkout 59f97b6
```

## 브런치 생성 및 체크아웃

```
git branch backup
git checkout backup
```

## merge

* add, commit, push후 merge
